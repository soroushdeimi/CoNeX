import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  DendriticSegmentModel,
  SegmentStorageModel,
  PredictiveSynapseModel,
  learnPrecision,
  PredictiveCodingNetwork,
} from '../assets/js/models/learning_numenta.js';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, 'golden', 'learning_numenta.json'), 'utf8'));

const REL = 1e-4;
const ABS = 1e-5;

function close(a, b, { rel = REL, abs = ABS } = {}) {
  assert.ok(
    Math.abs(a - b) <= abs + rel * Math.abs(b),
    `expected ${a} to be within ${rel} (abs ${abs}) of ${b}, diff = ${Math.abs(a - b)}`,
  );
}

function closeVec(a, b, opts) {
  assert.equal(a.length, b.length, `vector length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) close(a[i], b[i], opts);
}

// ---------------------------------------------------------------------------
// 1. DendriticSegmentModel
// ---------------------------------------------------------------------------

test('dendritic segment: step activations and permanences match golden', () => {
  const g = golden.dendritic_segments;
  const model = new DendriticSegmentModel({
    nCells: g.params.n_cells,
    nSegments: g.params.n_segments,
    nSynapsesPerSegment: g.params.n_synapses_per_segment,
    activationThreshold: g.params.activation_threshold,
    learningThreshold: g.params.learning_threshold,
    permanenceIncrement: g.params.permanence_increment,
    permanenceDecrement: g.params.permanence_decrement,
    initialPermanence: g.params.initial_permanence,
    connectedPermanence: g.params.connected_permanence,
    maxNewSynapses: g.params.max_new_synapses,
  });

  const step0 = g.steps[0];
  for (let s = 0; s < 2; s++) {
    model.setSegment(s, step0.targets_before[s], step0.permanences_before[s]);
  }

  for (const stepData of g.steps) {
    const actRes = model.computeActivation(stepData.presynaptic_activity);
    closeVec(actRes.activation, stepData.segment_activation);
    assert.deepEqual(actRes.active, stepData.segment_active);
    assert.deepEqual(actRes.matching, stepData.matching_segments);
    close(actRes.depolarization[0], stepData.depolarization);

    const grownTargets = stepData.targets_before[0].includes(-1)
      ? stepData.targets_after[0].slice(15)
      : null;

    model.learnOnSegments({
      activeCells: [true],
      winnerCells: [true],
      presynapticActivity: stepData.presynaptic_activity,
      newTargets: grownTargets,
    });

    for (let s = 0; s < 2; s++) {
      const seg = model.getSegment(s);
      closeVec(seg.permanences, stepData.permanences_after[s]);
      assert.deepEqual(seg.targets, stepData.targets_after[s]);
    }
  }
});

test('dendritic segment: parameter validations throw with bad value', () => {
  assert.throws(
    () => new DendriticSegmentModel({ nCells: 0 }),
    /nCells must be a positive integer, got 0/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ nSegments: -1 }),
    /nSegments must be a positive integer, got -1/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ nSynapsesPerSegment: 1.5 }),
    /nSynapsesPerSegment must be a positive integer, got 1.5/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ activationThreshold: -2 }),
    /activationThreshold must be a non-negative integer, got -2/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ learningThreshold: -1 }),
    /learningThreshold must be a non-negative integer, got -1/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ permanenceIncrement: 0 }),
    /permanenceIncrement must be in \(0, 1\], got 0/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ permanenceDecrement: 1.5 }),
    /permanenceDecrement must be in \(0, 1\], got 1.5/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ initialPermanence: -0.1 }),
    /initialPermanence must be in \[0, 1\], got -0.1/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ connectedPermanence: 2.0 }),
    /connectedPermanence must be in \[0, 1\], got 2/,
  );
  assert.throws(
    () => new DendriticSegmentModel({ maxNewSynapses: -5 }),
    /maxNewSynapses must be a non-negative integer, got -5/,
  );
});

// ---------------------------------------------------------------------------
// 2. SegmentStorageModel (TemporalMemory adapt, punish, grow)
// ---------------------------------------------------------------------------

test('temporal memory: adaptSegment reinforces active and punishes inactive', () => {
  const g = golden.temporal_memory_segments.adapt;
  const storage = new SegmentStorageModel(4);
  storage.setSegment(g.cell_idx, g.segment_idx, g.initial_targets, g.initial_perms);

  const actions = storage.adaptSegment(g.cell_idx, g.segment_idx, g.active_cells, g.perm_inc, g.perm_dec);
  assert.equal(actions.length, 3);
  assert.equal(actions[0].action, 'reinforced');
  assert.equal(actions[1].action, 'punished');
  assert.equal(actions[2].action, 'reinforced');

  const seg = storage.getSegment(g.cell_idx, g.segment_idx);
  closeVec(Array.from(seg.permanences), g.expected_perms);
});

test('temporal memory: punishSegments decrements active synapses of false predictions', () => {
  const g = golden.temporal_memory_segments.punish;
  const storage = new SegmentStorageModel(4);
  storage.setSegment(g.cell_idx, g.segment_idx, g.initial_targets, g.initial_perms);

  const punished = storage.punishSegments(
    g.cell_idx,
    g.active_cells,
    g.decrement,
    g.threshold,
    g.connected_perm,
  );
  assert.equal(punished.length, 2);

  const seg = storage.getSegment(g.cell_idx, g.segment_idx);
  closeVec(Array.from(seg.permanences), g.expected_perms);
});

test('temporal memory: growSynapses connects to candidate winner cells', () => {
  const g = golden.temporal_memory_segments.grow;
  const storage = new SegmentStorageModel(4);
  storage.setSegment(g.cell_idx, g.segment_idx, [1, 2, 3], [0.35, 0.13, 0.35]);

  const grown = storage.growSynapses(
    g.cell_idx,
    g.segment_idx,
    g.winner_cells,
    g.initial_perm,
    g.max_new,
    g.max_synapses,
  );
  assert.equal(grown.length, 1);
  assert.equal(grown[0].target, 0);

  const seg = storage.getSegment(g.cell_idx, g.segment_idx);
  assert.deepEqual(Array.from(seg.targets), [1, 2, 3, 0]);
  closeVec(Array.from(seg.permanences), [0.35, 0.13, 0.35, 0.21]);
});

test('temporal memory: SegmentStorageModel rejects bad nCells', () => {
  assert.throws(() => new SegmentStorageModel(0), /nCells must be positive integer, got 0/);
  assert.throws(() => new SegmentStorageModel(-3), /nCells must be positive integer, got -3/);
});

// ---------------------------------------------------------------------------
// 3. PredictiveSynapseModel
// ---------------------------------------------------------------------------

test('predictive synapse: error_driven multi-step trajectory matches golden', () => {
  const g = golden.predictive_synapse;
  const p = g.params;
  const model = new PredictiveSynapseModel({
    srcSize: p.src_size,
    dstSize: p.dst_size,
    learningRate: p.learning_rate,
    dt: p.dt,
    learningType: 'error_driven',
    initialWeights: p.initial_weights,
  });

  const expected = g.error_driven;
  for (let step = 0; step < expected.weights_trace.length; step++) {
    const res = model.step({
      srcActivity: p.src_activity,
      dstError: p.dst_error,
      dstActivity: p.dst_activity,
    });
    closeVec(res.weights, expected.weights_trace[step]);
    closeVec(res.eligibility, expected.eligibility_trace[step]);
  }
});

test('predictive synapse: hebbian multi-step trajectory matches golden', () => {
  const g = golden.predictive_synapse;
  const p = g.params;
  const model = new PredictiveSynapseModel({
    srcSize: p.src_size,
    dstSize: p.dst_size,
    learningRate: p.learning_rate,
    dt: p.dt,
    learningType: 'hebbian',
    initialWeights: p.initial_weights,
  });

  const expected = g.hebbian;
  for (let step = 0; step < expected.weights_trace.length; step++) {
    const res = model.step({
      srcActivity: p.src_activity,
      dstError: p.dst_error,
      dstActivity: p.dst_activity,
    });
    closeVec(res.weights, expected.weights_trace[step]);
    closeVec(res.eligibility, expected.eligibility_trace[step]);
  }
});

test('predictive synapse: combined multi-step trajectory matches golden', () => {
  const g = golden.predictive_synapse;
  const p = g.params;
  const model = new PredictiveSynapseModel({
    srcSize: p.src_size,
    dstSize: p.dst_size,
    learningRate: p.learning_rate,
    dt: p.dt,
    learningType: 'combined',
    initialWeights: p.initial_weights,
  });

  const expected = g.combined;
  for (let step = 0; step < expected.weights_trace.length; step++) {
    const res = model.step({
      srcActivity: p.src_activity,
      dstError: p.dst_error,
      dstActivity: p.dst_activity,
    });
    closeVec(res.weights, expected.weights_trace[step]);
    closeVec(res.eligibility, expected.eligibility_trace[step]);
  }
});

test('predictive synapse: invalid parameters throw with bad value', () => {
  assert.throws(() => new PredictiveSynapseModel({ srcSize: 0 }), /srcSize must be a positive integer, got 0/);
  assert.throws(() => new PredictiveSynapseModel({ dstSize: -2 }), /dstSize must be a positive integer, got -2/);
  assert.throws(() => new PredictiveSynapseModel({ learningRate: 0 }), /learningRate must be positive, got 0/);
  assert.throws(() => new PredictiveSynapseModel({ dt: -1 }), /dt must be positive, got -1/);
  assert.throws(
    () => new PredictiveSynapseModel({ learningType: 'unknown_rule' }),
    /learningType must be one of error_driven, hebbian, combined, got unknown_rule/,
  );
});

// ---------------------------------------------------------------------------
// 4. Precision rule learnPrecision
// ---------------------------------------------------------------------------

test('precision rule: gradient and updated precision match golden', () => {
  const g = golden.precision_rule;
  const p = g.params;
  const res = learnPrecision({
    sensoryPrecision: p.initial_sensory_precision,
    signedError: p.signed_error,
    precisionLearningRate: p.precision_learning_rate,
    minPrecision: p.min_precision,
    maxPrecision: p.max_precision,
    dt: p.dt,
  });

  closeVec(res.precisionGrad, g.precision_grad);
  closeVec(res.sensoryPrecision, g.updated_sensory_precision);
});

test('precision rule: invalid parameters throw with bad value', () => {
  assert.throws(
    () => learnPrecision({ sensoryPrecision: [1], signedError: [0], precisionLearningRate: 0 }),
    /precisionLearningRate must be positive, got 0/,
  );
  assert.throws(
    () => learnPrecision({ sensoryPrecision: [1], signedError: [0], minPrecision: -0.1 }),
    /minPrecision must be positive, got -0.1/,
  );
  assert.throws(
    () => learnPrecision({ sensoryPrecision: [1], signedError: [0], minPrecision: 10, maxPrecision: 5 }),
    /maxPrecision must be >= minPrecision, got 5/,
  );
});

// ---------------------------------------------------------------------------
// 5. PredictiveCodingNetwork
// ---------------------------------------------------------------------------

test('predictive network: 20-step learning trajectory matches golden', () => {
  const g = golden.predictive_network;
  const net = new PredictiveCodingNetwork(g.size, {
    learningRate: g.params.learning_rate,
    precisionTau: g.params.precision_tau,
    precisionLearningRate: g.params.precision_learning_rate,
    minPrecision: g.params.min_precision,
    maxPrecision: g.params.max_precision,
    errorNonlinearity: g.params.error_nonlinearity,
    useSparseErrors: g.params.use_sparse_errors,
  });

  net.activity = Float64Array.from(g.sensory);
  net.topDownInput = Float64Array.from(g.higher);

  for (let s = 0; s < g.steps; s++) {
    const res = net.step();
    const exp = g.trace[s];

    close(res.meanAbsError, exp.mean_abs_error, { rel: 1e-3, abs: 1e-4 });
    close(res.meanPrecision, exp.mean_precision, { rel: 1e-3, abs: 1e-4 });
    close(res.freeEnergy, exp.free_energy, { rel: 1e-3, abs: 1e-3 });
    closeVec(res.prediction, exp.prediction, { rel: 1e-3, abs: 1e-4 });
    closeVec(res.signedError, exp.signed_error, { rel: 1e-3, abs: 1e-4 });
    closeVec(res.precision, exp.precision, { rel: 1e-3, abs: 1e-4 });
    closeVec(res.weights, exp.top_down_weights, { rel: 1e-3, abs: 1e-4 });
  }
});

test('predictive network: prediction error monotonically decreases during learning', () => {
  const g = golden.predictive_network;
  const net = new PredictiveCodingNetwork(g.size, {
    learningRate: g.params.learning_rate,
    precisionTau: g.params.precision_tau,
    precisionLearningRate: g.params.precision_learning_rate,
  });

  net.activity = Float64Array.from(g.sensory);
  net.topDownInput = Float64Array.from(g.higher);

  let prevError = Infinity;
  for (let s = 0; s < 10; s++) {
    const res = net.step();
    assert.ok(res.meanAbsError < prevError, `step ${s}: error did not decrease (${res.meanAbsError} vs ${prevError})`);
    prevError = res.meanAbsError;
  }
});

test('predictive network: rejects invalid size', () => {
  assert.throws(() => new PredictiveCodingNetwork(0), /size must be positive integer, got 0/);
  assert.throws(() => new PredictiveCodingNetwork(-1), /size must be positive integer, got -1/);
});
