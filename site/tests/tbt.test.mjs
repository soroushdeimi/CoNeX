import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  GridCellModule,
  DisplacementCellModule,
  LocationEncoder,
  DendriticSegment,
  ActiveDendriteComputation,
  makeDendriteState,
  ContextualPrediction,
  ColumnVoting,
  ConsensusNetwork,
  SDROverlap,
  PredictiveCodingLevel,
  HierarchicalPredictiveCoding,
  corticalColumnStructure,
} from '../assets/js/models/tbt.js';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, 'golden', 'tbt.json'), 'utf8'));

// Tolerances: JS uses float64, CoNeX uses float32, so allow a small relative gap.
const REL = 1e-4;
const ABS = 1e-5;

function close(a, b, { rel = REL, abs = ABS } = {}) {
  assert.ok(
    Math.abs(a - b) <= abs + rel * Math.abs(b),
    `expected ${a} to be within ${rel} of ${b}`,
  );
}

function closeVec(a, b, opts) {
  assert.equal(a.length, b.length, 'vector length mismatch');
  for (let i = 0; i < a.length; i++) close(a[i], b[i], opts);
}

// ---------------------------------------------------------------------------
// GridCellModule
// ---------------------------------------------------------------------------

test('grid: scales match CoNeX logspace', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  closeVec(Array.from(m.scales), g.scales);
});

test('grid: orientations match CoNeX linspace', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  closeVec(Array.from(m.orientations), g.orientations);
});

test('grid: wave vectors match CoNeX', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  closeVec(Array.from(m.waveVectors), g.wave_vectors.flat());
});

test('grid: activation at each position matches CoNeX', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  for (let p = 0; p < g.positions.length; p++) {
    const act = m.computeActivation(g.positions[p]);
    closeVec(Array.from(act), g.activations[p]);
  }
});

test('grid: path integration accumulates velocity', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  for (const v of g.velocity_steps) m.pathIntegrate(v, g.dt);
  close(m.gridPosition[0], g.position_after_integration[0]);
  close(m.gridPosition[1], g.position_after_integration[1]);
});

test('grid: hexagonal lattice periodicity', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  const origin = m.computeActivation([0, 0]).slice(0, g.cells_per_module);
  const shifted = m.computeActivation(g.periodicity.step).slice(0, g.cells_per_module);
  closeVec(Array.from(origin), g.periodicity.origin, { abs: 1e-4 });
  closeVec(Array.from(shifted), g.periodicity.shifted, { abs: 1e-4 });
});

test('grid: activation is maximal at lattice points and lower between them', () => {
  const g = golden.grid;
  const m = new GridCellModule({
    nModules: g.n_modules,
    cellsPerModule: g.cells_per_module,
    scaleRange: g.scale_range,
    orientationRange: g.orientation_range,
    phases: g.phases.flat(),
  });
  // Module 0, cell 0. The hexagonal lattice fundamental translations are
  // a1 = (2*lambda/sqrt(3)) * (cos(theta+30deg), sin(theta+30deg))
  // a2 = a1 rotated by 60deg.
  const lam = m.scales[0];
  const th = m.orientations[0];
  const px = m.gridPhase[0];
  const py = m.gridPhase[1];
  const L = (2 * lam) / Math.sqrt(3);
  const a1 = [L * Math.cos(th + Math.PI / 6), L * Math.sin(th + Math.PI / 6)];
  const a2 = [L * Math.cos(th + Math.PI / 2), L * Math.sin(th + Math.PI / 2)];
  const act = (x, y) => m.computeActivation([x, y])[0];

  // The phase point is a lattice maximum (all three cosines = 1).
  const atPhase = act(px, py);
  assert.ok(atPhase > 0.99, `phase point should be a maximum, got ${atPhase}`);

  // Every lattice point p = i*a1 + j*a2 is also a maximum.
  for (let i = -2; i <= 2; i++) {
    for (let j = -2; j <= 2; j++) {
      const x = px + i * a1[0] + j * a2[0];
      const y = py + i * a1[1] + j * a2[1];
      close(act(x, y), atPhase, { abs: 1e-4 });
    }
  }

  // Midpoints between lattice points are strictly lower (the "valleys").
  const mid12 = act(px + (a1[0] + a2[0]) / 2, py + (a1[1] + a2[1]) / 2);
  const mid1 = act(px + a1[0] / 2, py + a1[1] / 2);
  assert.ok(mid12 < atPhase - 0.3, `midpoint should be a valley, got ${mid12}`);
  assert.ok(mid1 < atPhase - 0.3, `half-step should be a valley, got ${mid1}`);
});

test('grid: missing phases throws', () => {
  assert.throws(
    () => new GridCellModule({ nModules: 2, cellsPerModule: 5 }),
    /phases/,
  );
});

test('grid: wrong phases length throws', () => {
  assert.throws(
    () => new GridCellModule({ nModules: 2, cellsPerModule: 5, phases: [0, 1] }),
    /phases length/,
  );
});

// ---------------------------------------------------------------------------
// DisplacementCellModule
// ---------------------------------------------------------------------------

test('displacement: preferred displacements match CoNeX', () => {
  const g = golden.displacement;
  const m = new DisplacementCellModule({
    nCells: g.n_cells,
    maxDisplacement: g.max_displacement,
    sigma: g.sigma,
  });
  closeVec(Array.from(m.preferredDx), g.preferred_dx);
  closeVec(Array.from(m.preferredDy), g.preferred_dy);
});

test('displacement: encode + decode round trip matches CoNeX', () => {
  const g = golden.displacement;
  const m = new DisplacementCellModule({
    nCells: g.n_cells,
    maxDisplacement: g.max_displacement,
    sigma: g.sigma,
  });
  for (const key of ['pair1', 'pair2']) {
    const act = m.computeDisplacement(g[key].from, g[key].to);
    closeVec(Array.from(act), g[key].activation);
    const decoded = m.decodeDisplacement(act);
    close(decoded[0], g[key].decoded[0]);
    close(decoded[1], g[key].decoded[1]);
  }
});

test('displacement: decode of empty activation is zero', () => {
  const m = new DisplacementCellModule({ nCells: 100 });
  const decoded = m.decodeDisplacement(new Float64Array(100));
  assert.deepEqual(decoded, [0, 0]);
});

// ---------------------------------------------------------------------------
// LocationEncoder
// ---------------------------------------------------------------------------

test('location: winner-take-all picks the top-n grid cells', () => {
  const m = new LocationEncoder({ locationSize: 16, gridWeight: 0.5, sparsity: 0.2 });
  const grid = new Float64Array(16);
  grid[3] = 0.9;
  grid[7] = 0.8;
  grid[11] = 0.7;
  const sparse = m.encodeLocation(grid, null);
  assert.equal(sparse[3], 1);
  assert.equal(sparse[7], 1);
  assert.equal(sparse[11], 1);
  assert.equal(sparse.reduce((a, b) => a + b, 0), 3);
  assert.deepEqual(Array.from(m.activeLocations), [3, 7, 11]);
});

test('location: zero input returns the combined (all zero) vector', () => {
  const m = new LocationEncoder({ locationSize: 8, gridWeight: 0.5, sparsity: 0.25 });
  const out = m.encodeLocation(null, null);
  assert.equal(out.length, 8);
  assert.equal(out.reduce((a, b) => a + b, 0), 0);
});

// ---------------------------------------------------------------------------
// DendriticSegment
// ---------------------------------------------------------------------------

test('dendrites: segment activation and depolarization match CoNeX', () => {
  const g = golden.dendrites.segment;
  const m = new DendriticSegment({
    nSegments: g.n_segments,
    nCells: g.n_cells,
    nSynapsesPerSegment: g.n_synapses,
    activationThreshold: g.activation_threshold,
    learningThreshold: g.learning_threshold,
    connectedPermanence: g.connected_permanence,
  });
  for (let slot = 0; slot < g.targets.length; slot++) {
    m.segmentTargets[slot] = g.targets[slot];
    m.segmentPermanences[slot] = g.permanence;
  }
  const activeAll = new Array(g.n_cells).fill(false);
  for (const t of g.targets) activeAll[t] = true;
  m.computeSegmentActivation(activeAll);
  m.computeDepolarization();
  closeVec(Array.from(m.segmentActivation), g.all_active.segment_activation);
  assert.deepEqual(Array.from(m.segmentActive, Boolean), g.all_active.segment_active);
  closeVec(Array.from(m.depolarization), g.all_active.depolarization);

  const activeOne = new Array(g.n_cells).fill(false);
  activeOne[2] = true;
  m.computeSegmentActivation(activeOne);
  m.computeDepolarization();
  closeVec(Array.from(m.segmentActivation), g.one_active.segment_activation);
  assert.deepEqual(Array.from(m.matchingSegments, Boolean), g.one_active.matching);
  closeVec(Array.from(m.depolarization), g.one_active.depolarization);
});

test('dendrites: unconnected synapses do not count', () => {
  const m = new DendriticSegment({
    nSegments: 1,
    nCells: 4,
    nSynapsesPerSegment: 3,
    activationThreshold: 2,
    learningThreshold: 1,
    connectedPermanence: 0.5,
  });
  m.segmentTargets[0] = 1;
  m.segmentPermanences[0] = 0.1; // below connected permanence
  const active = [false, true, false, false];
  m.computeSegmentActivation(active);
  assert.equal(m.segmentActivation[0], 0);
});

// ---------------------------------------------------------------------------
// ActiveDendriteComputation
// ---------------------------------------------------------------------------

test('dendrites: NMDA nonlinearity matches CoNeX', () => {
  const g = golden.dendrites;
  const m = new ActiveDendriteComputation({ nmdaThreshold: 0.5, nmdaGain: 2.0 });
  close(m.nmdaNonlinearity(0.1), g.nmda.low[0]);
  close(m.nmdaNonlinearity(1.0), g.nmda.high[0]);
});

test('dendrites: integration over 20 steps matches CoNeX', () => {
  const g = golden.dendrites.integration;
  const m = new ActiveDendriteComputation({ nmdaThreshold: 0.5, nmdaGain: 2.0 });
  const state = makeDendriteState();
  let current = 0;
  for (let i = 0; i < g.steps; i++) {
    current = m.computeDendriticIntegration(state, 1.0, g.proximal_input, g.distal_input);
  }
  close(state.proximal, g.proximal_potential[0]);
  close(state.distal, g.distal_potential[0]);
  close(state.calcium, g.dendritic_calcium[0]);
  close(current, g.total_current[0]);
  assert.equal(state.dendriticSpike, g.dendritic_spike[0] ? 1 : 0);
});

test('dendrites: backprop raises calcium', () => {
  const m = new ActiveDendriteComputation({ backpropStrength: 0.3 });
  const state = makeDendriteState();
  const before = state.calcium;
  m.applyBackprop(state, 1);
  assert.ok(state.calcium > before);
});

// ---------------------------------------------------------------------------
// ContextualPrediction
// ---------------------------------------------------------------------------

test('dendrites: context makes cells predictive and unpredicted cells burst', () => {
  const g = golden.dendrites.context;
  const m = new ContextualPrediction({
    nCells: 8,
    depolarizationThreshold: g.depolarization_threshold,
    predictionWindow: g.prediction_window,
    burstThreshold: g.burst_threshold,
  });
  m.updatePredictions(g.drive);
  assert.deepEqual(Array.from(m.predictive, Boolean), g.predictive);
  const { activation, bursting } = m.processActivation(new Array(8).fill(1));
  closeVec(Array.from(activation), g.activation);
  assert.deepEqual(Array.from(bursting, Boolean), g.bursting);
});

test('dendrites: predictions expire after the window', () => {
  const m = new ContextualPrediction({
    nCells: 4,
    depolarizationThreshold: 0.5,
    predictionWindow: 3,
    burstThreshold: 0.3,
  });
  const drive = [0.9, 0, 0, 0];
  m.updatePredictions(drive);
  assert.equal(m.predictive[0], 1);
  for (let i = 0; i < 4; i++) m.updatePredictions([0, 0, 0, 0]);
  assert.equal(m.predictive[0], 0);
});

// ---------------------------------------------------------------------------
// ColumnVoting / ConsensusNetwork / SDROverlap
// ---------------------------------------------------------------------------

test('voting: full consensus run matches CoNeX', () => {
  const g = golden.voting;
  const memories = g.memories.map((row) => Float64Array.from(row));
  const columns = [];
  for (let c = 0; c < 3; c++) {
    columns.push(
      new ColumnVoting({
        nObjects: g.n_objects,
        representationSize: g.representation_size,
        sparsity: g.sparsity,
        votingThreshold: g.voting_threshold,
        decayRate: g.decay_rate,
        minVotesForConsensus: g.min_votes_for_consensus,
      }),
    );
  }
  const consensus = new ConsensusNetwork({
    convergenceThreshold: g.convergence_threshold,
  });

  for (let step = 0; step < g.steps.length; step++) {
    for (let c = 0; c < 3; c++) {
      const votes = columns[c].castVote(g.local_reps[c], memories);
      columns[c].receiveVotes(votes);
    }
    consensus.broadcastVotes(columns);
    for (const col of columns) col.forward();
    const reached = columns.map((col) => col.checkConsensus(3));
    const netConsensus = consensus.checkNetworkConsensus(columns);
    consensus.forward();

    const ref = g.steps[step];
    for (let c = 0; c < 3; c++) {
      closeVec(Array.from(columns[c].voteAccumulator), ref.accumulators[c]);
    }
    assert.deepEqual(reached, ref.consensus);
    assert.deepEqual(
      columns.map((col) => col.consensusObject),
      ref.consensus_objects,
    );
    assert.equal(netConsensus, ref.network_consensus);
  }
});

test('voting: cast_vote with no active bits returns all zeros', () => {
  const m = new ColumnVoting({ nObjects: 4, representationSize: 8, sparsity: 0.25 });
  const votes = m.castVote(new Float64Array(8), [Float64Array.from([1, 0, 1, 0, 0, 0, 0, 0])]);
  assert.deepEqual(Array.from(votes), [0, 0, 0, 0]);
});

test('voting: votes below threshold are zeroed', () => {
  const m = new ColumnVoting({
    nObjects: 2,
    representationSize: 8,
    sparsity: 0.25,
    votingThreshold: 0.5,
  });
  // local rep overlaps object 0 on 1 of 2 active bits -> 0.5 (kept), object 1 on 0 -> 0
  const local = Float64Array.from([1, 1, 0, 0, 0, 0, 0, 0]);
  const memories = [
    Float64Array.from([1, 0, 0, 0, 0, 0, 0, 0]),
    Float64Array.from([0, 0, 0, 0, 0, 0, 0, 0]),
  ];
  const votes = m.castVote(local, memories);
  close(votes[0], 0.5);
  assert.equal(votes[1], 0);
});

test('sdr: overlap, ratio and jaccard match CoNeX', () => {
  const g = golden.voting.sdr_metrics;
  const a = Float64Array.from(golden.voting.memories[0]);
  const b = Float64Array.from(golden.voting.memories[1]);
  assert.equal(SDROverlap.computeOverlap(a, b), g.overlap);
  close(SDROverlap.computeOverlapRatio(a, b), g.ratio);
  close(SDROverlap.computeJaccard(a, b), g.jaccard);
});

test('sdr: overlap ratio of empty SDR is zero', () => {
  assert.equal(SDROverlap.computeOverlapRatio(new Float64Array(4), Float64Array.from([1, 0, 0, 0])), 0);
});

// ---------------------------------------------------------------------------
// PredictiveCodingLevel
// ---------------------------------------------------------------------------

test('predictive: 40-step learning trace matches CoNeX', () => {
  const g = golden.predictive;
  const m = new PredictiveCodingLevel(g.size, {
    predictionTau: g.config.prediction_tau,
    errorTau: g.config.error_tau,
    precisionTau: g.config.precision_tau,
    learningRate: g.config.learning_rate,
    precisionLearningRate: g.config.precision_learning_rate,
    minPrecision: g.config.min_precision,
    maxPrecision: g.config.max_precision,
    errorNonlinearity: g.config.error_nonlinearity,
    useSparseErrors: g.config.use_sparse_errors,
    errorThreshold: g.config.error_threshold,
    temporalSmoothing: g.config.temporal_smoothing,
  });
  m.activity = Float64Array.from(g.sensory);
  m.topDownInput = Float64Array.from(g.higher_level);

  for (let i = 0; i < g.trace.length; i++) {
    m.forward();
    const ref = g.trace[i];
    const meanAbs = m.errorMagnitude.reduce((a, b) => a + b, 0) / m.size;
    const meanPrec = m.precision.reduce((a, b) => a + b, 0) / m.size;
    close(meanAbs, ref.mean_abs_error, { rel: 1e-3 });
    close(meanPrec, ref.mean_precision, { rel: 1e-3 });
    close(m.freeEnergy, ref.free_energy, { rel: 1e-3, abs: 1e-3 });
  }
});

test('predictive: perturbation raises the error again', () => {
  const g = golden.predictive;
  const m = new PredictiveCodingLevel(g.size, {
    errorNonlinearity: g.config.error_nonlinearity,
    useSparseErrors: g.config.use_sparse_errors,
    learningRate: g.config.learning_rate,
    precisionTau: g.config.precision_tau,
    precisionLearningRate: g.config.precision_learning_rate,
  });
  m.activity = Float64Array.from(g.sensory);
  m.topDownInput = Float64Array.from(g.higher_level);
  for (let i = 0; i < g.steps; i++) m.forward();
  const settled = m.errorMagnitude.reduce((a, b) => a + b, 0) / m.size;
  assert.ok(settled < 1e-3, 'error should have settled before perturbation');
});

test('predictive: relu nonlinearity clips negative error', () => {
  const m = new PredictiveCodingLevel(2, { errorNonlinearity: 'relu', useSparseErrors: false });
  m.activity = Float64Array.from([0.0, 1.0]);
  m.topDownInput = Float64Array.from([0.0, 1.0]);
  m.topDown();
  m.error();
  // prediction = 0.5 * top_down = [0, 0.5]; signed error = [0, 0.5]; relu keeps both
  assert.equal(m.predictionError[0], 0);
  close(m.predictionError[1], 0.5);
});

// ---------------------------------------------------------------------------
// HierarchicalPredictiveCoding
// ---------------------------------------------------------------------------

test('hierarchy: one forward pass matches CoNeX (incl. the 2-level quirk)', () => {
  const g = golden.hierarchy;
  const m = new HierarchicalPredictiveCoding(
    g.level_sizes,
    [Float64Array.from(g.feedforward_weights)],
    [Float64Array.from(g.feedback_weights)],
  );
  m.forward(Float64Array.from(g.sensory));
  for (let i = 0; i < g.level_sizes.length; i++) {
    closeVec(Array.from(m.levelPredictions[i]), g.level_predictions[i]);
    closeVec(Array.from(m.levelErrors[i]), g.level_errors[i]);
  }
});

// ---------------------------------------------------------------------------
// Cortical column structure
// ---------------------------------------------------------------------------

test('structure: layer sizes and connections match CoNeX', () => {
  const g = golden.structure;
  const s = corticalColumnStructure(g.column_size);
  assert.deepEqual(s.layerSizes, g.layer_sizes);
  assert.deepEqual(
    s.connections.map((c) => `${c.from}_to_${c.to}`).sort(),
    g.connections,
  );
});

// ---------------------------------------------------------------------------
// tbt-grid hero mappings (model state -> drawn quantity / narration)
// ---------------------------------------------------------------------------

import {
  trailAlpha,
  glowRadius,
  gridNarration,
} from '../assets/plates/tbt-grid.js';
import {
  arrowLength,
  cellGlow,
  displacementNarration,
} from '../assets/plates/tbt-displacement.js';

test('grid hero: trail alpha fades linearly from 1 to 0 over the window', () => {
  assert.equal(trailAlpha(0, 5), 1);
  assert.equal(trailAlpha(2.5, 5), 0.5);
  assert.equal(trailAlpha(5, 5), 0);
  // clamped: negative age -> 1, age past window -> 0
  assert.equal(trailAlpha(-1, 5), 1);
  assert.equal(trailAlpha(9, 5), 0);
  // zero window -> no trail
  assert.equal(trailAlpha(0, 0), 0);
});

test('grid hero: glow radius grows with mean activation and is bounded', () => {
  assert.equal(glowRadius(0), 10);
  assert.equal(glowRadius(1), 40);
  assert.equal(glowRadius(0.5), 25);
  // clamped to the [0,1] activation range
  assert.equal(glowRadius(-3), 10);
  assert.equal(glowRadius(7), 40);
});

test('grid hero: narration reports the walk speed and firing level', () => {
  assert.equal(
    gridNarration({ meanAct: 0.5, speed: 0 }),
    'agent at rest: the grid cells hold the position code',
  );
  const high = gridNarration({ meanAct: 0.9, speed: 0.6 });
  assert.match(high, /walks at 0\.60 a\.u\.\/s/);
  assert.match(high, /high level/);
  assert.match(high, /mean 0\.90 a\.u\./);
  const low = gridNarration({ meanAct: 0.1, speed: 0.2 });
  assert.match(low, /low level/);
  const mid = gridNarration({ meanAct: 0.5, speed: 0.2 });
  assert.match(mid, /medium level/);
});

// ---------------------------------------------------------------------------
// tbt-displacement hero mappings
// ---------------------------------------------------------------------------

test('displacement hero: arrow length scales with magnitude and is bounded', () => {
  assert.equal(arrowLength(0, 10), 0);
  assert.equal(arrowLength(10, 10), 120);
  assert.equal(arrowLength(5, 10), 60);
  // clamped: magnitude above max -> full length, negative -> 0
  assert.equal(arrowLength(20, 10), 120);
  assert.equal(arrowLength(-4, 10), 0);
  // zero max displacement -> no arrow
  assert.equal(arrowLength(5, 0), 0);
});

test('displacement hero: cell glow is the clamped activation', () => {
  assert.equal(cellGlow(0), 0);
  assert.equal(cellGlow(1), 1);
  assert.equal(cellGlow(0.4), 0.4);
  assert.equal(cellGlow(-0.2), 0);
  assert.equal(cellGlow(1.7), 1);
});

test('displacement hero: narration reports distance and decoded vector', () => {
  const line = displacementNarration({
    magnitude: 3.5,
    decoded: [2.0, 2.5],
    maxDisplacement: 10,
  });
  assert.match(line, /3\.50 a\.u\. from A/);
  assert.match(line, /\(2\.00, 2\.50\) a\.u\./);
  assert.match(line, /max 10\.0 a\.u\./);
});


// ---------------------------------------------------------------------------
// tbt-dendrites hero mappings
// ---------------------------------------------------------------------------

import {
  gaugeFill,
  segmentState,
  isPredictive,
  dendriteNarration,
  cellLabel,
} from '../assets/plates/tbt-dendrites.js';

test('dendrites hero: gauge fill is activation over threshold, clamped', () => {
  assert.equal(gaugeFill(0, 6), 0);
  assert.equal(gaugeFill(6, 6), 1);
  assert.equal(gaugeFill(3, 6), 0.5);
  assert.equal(gaugeFill(12, 6), 1); // clamped above
  assert.equal(gaugeFill(3, 0), 0); // zero threshold -> no fill
});

test('dendrites hero: segment state is active / matching / below', () => {
  assert.equal(segmentState(6, 6, 4), 'active');
  assert.equal(segmentState(5, 6, 4), 'matching');
  assert.equal(segmentState(3, 6, 4), 'below');
  // boundary: exactly at the learning threshold is matching
  assert.equal(segmentState(4, 6, 4), 'matching');
  // boundary: exactly at the activation threshold is active
  assert.equal(segmentState(6, 6, 4), 'active');
});

test('dendrites hero: a cell is predictive at and above the threshold', () => {
  assert.equal(isPredictive(0.5, 0.5), true);
  assert.equal(isPredictive(0.9, 0.5), true);
  assert.equal(isPredictive(0.4, 0.5), false);
});

test('dendrites hero: the printed cell label is a rounded integer', () => {
  // a mid-ease value must print as an integer, not the raw float
  assert.equal(cellLabel(2.49985000019803), '2');
  assert.equal(cellLabel(1.49985000019803), '1');
  assert.equal(cellLabel(0), '0');
  assert.equal(cellLabel(3), '3');
  // rounds to nearest
  assert.equal(cellLabel(2.5), '3');
  assert.equal(cellLabel(1.4), '1');
});

test('dendrites hero: narration reports bits arrived and predictive count', () => {
  assert.equal(dendriteNarration({ k: 0, nActive: 6, nPredictive: 0 }), 'no context yet: all segments are silent');
  const one = dendriteNarration({ k: 1, nActive: 6, nPredictive: 1 });
  assert.match(one, /1 of 6 context bits arrived/);
  assert.match(one, /1 cell now predictive/);
  const many = dendriteNarration({ k: 4, nActive: 6, nPredictive: 3 });
  assert.match(many, /4 of 6 context bits arrived/);
  assert.match(many, /3 cells now predictive/);
});

test('dendrites hero: narration floors the fractional bit count to an integer', () => {
  // the player time is fractional; the narration must show an integer count
  const line = dendriteNarration({ k: 3.499799999999987, nActive: 6, nPredictive: 2 });
  assert.match(line, /3 of 6 context bits arrived/);
  assert.doesNotMatch(line, /3\.49/);
  // just under a whole number still floors down
  assert.match(dendriteNarration({ k: 0.999, nActive: 6, nPredictive: 0 }), /no context yet/);
  assert.match(dendriteNarration({ k: 1.0, nActive: 6, nPredictive: 0 }), /1 of 6 context bits arrived/);
});

// ---------------------------------------------------------------------------
// tbt-voting hero mappings
// ---------------------------------------------------------------------------

import {
  barFraction,
  topObject,
  consensusObject,
  votingNarration,
} from '../assets/plates/tbt-voting.js';

test('voting hero: bar fraction is value over max, clamped', () => {
  assert.equal(barFraction(0, 4), 0);
  assert.equal(barFraction(4, 4), 1);
  assert.equal(barFraction(2, 4), 0.5);
  assert.equal(barFraction(8, 4), 1); // clamped above
  assert.equal(barFraction(2, 0), 0); // zero max -> no bar
});

test('voting hero: top object is the highest accumulator', () => {
  assert.equal(topObject([0.1, 0.9, 0.3]), 1);
  assert.equal(topObject([0.5, 0.5, 0.5]), 0); // ties -> lowest index
  assert.equal(topObject([0, 0, 0]), 0);
});

test('voting hero: consensus object is the agreed object or -1', () => {
  assert.equal(consensusObject([2, 2, 2]), 2);
  assert.equal(consensusObject([2, 2, -1]), 2); // -1 means no column consensus
  assert.equal(consensusObject([1, 2, 3]), 1); // no agreement -> first (count 1)
  assert.equal(consensusObject([-1, -1, -1]), -1);
});

test('voting hero: narration reports consensus state', () => {
  // every column shares the same top candidate -> consensus with the step count
  assert.equal(votingNarration({ step: 3, objs: [2, 2, 2], net: true }), 'consensus: object 2 after 3 steps');
  assert.equal(votingNarration({ step: 1, objs: [2, 2, -1], net: false }), 'step 1: columns are converging on object 2');
  assert.equal(votingNarration({ step: 0, objs: [-1, -1, -1], net: false }), 'step 0: no consensus yet');
});

// ---------------------------------------------------------------------------
// tbt-predictive hero mappings
// ---------------------------------------------------------------------------

import {
  particleRate,
  precisionKnob,
  predictiveNarration,
} from '../assets/plates/tbt-predictive.js';

test('predictive hero: particle rate is proportional to magnitude, clamped', () => {
  assert.equal(particleRate(0, 1, 14), 0);
  assert.equal(particleRate(1, 1, 14), 14);
  assert.equal(particleRate(0.5, 1, 14), 7);
  assert.equal(particleRate(2, 1, 14), 14); // clamped above
  assert.equal(particleRate(0.5, 0, 14), 0); // zero max -> no particles
});

test('predictive hero: precision knob maps precision to [0,1]', () => {
  assert.equal(precisionKnob(0, 0, 100), 0);
  assert.equal(precisionKnob(100, 0, 100), 1);
  assert.equal(precisionKnob(50, 0, 100), 0.5);
  assert.equal(precisionKnob(200, 0, 100), 1); // clamped above
  assert.equal(precisionKnob(-5, 0, 100), 0); // clamped below
  assert.equal(precisionKnob(5, 1, 1), 0); // degenerate range
});

test('predictive hero: narration reports the error trend and values', () => {
  const small = predictiveNarration({ meanAbs: 0.01, freeEnergy: 0.5, meanPrec: 2 });
  assert.match(small, /error is small/);
  assert.match(small, /mean \|error\| 0\.010 a\.u\./);
  assert.match(small, /free energy 0\.500 a\.u\./);
  const large = predictiveNarration({ meanAbs: 0.4, freeEnergy: 3.2, meanPrec: 1 });
  assert.match(large, /error is large/);
});

// ---------------------------------------------------------------------------
// struct-column hero mappings
// ---------------------------------------------------------------------------

import {
  pulsePosition,
  columnNarration,
} from '../assets/plates/struct-column.js';

const CONN_ORDER = ['L4_to_L23', 'L23_to_L4', 'L23_to_L5', 'L6_to_L4'];

test('column hero: pulse position walks the connections in CoNeX order', () => {
  // cycle 0 -> first connection, fraction 0
  assert.deepEqual(pulsePosition(0, CONN_ORDER), { key: 'L4_to_L23', fraction: 0 });
  // a quarter of the way -> second connection, fraction 0
  assert.deepEqual(pulsePosition(0.25, CONN_ORDER), { key: 'L23_to_L4', fraction: 0 });
  // halfway through the second connection
  const mid = pulsePosition(0.375, CONN_ORDER);
  assert.equal(mid.key, 'L23_to_L4');
  assert.equal(mid.fraction, 0.5);
  // just before the end -> last connection, fraction near 1
  const end = pulsePosition(0.999, CONN_ORDER);
  assert.equal(end.key, 'L6_to_L4');
  assert.ok(end.fraction > 0.9);
});

test('column hero: pulse position wraps the cycle', () => {
  // cycle 1.0 wraps back to the first connection
  assert.deepEqual(pulsePosition(1.0, CONN_ORDER), { key: 'L4_to_L23', fraction: 0 });
  // cycle 1.25 wraps to the second connection
  assert.equal(pulsePosition(1.25, CONN_ORDER).key, 'L23_to_L4');
});

test('column hero: narration names the pulse and its connection', () => {
  assert.equal(columnNarration({ conn: 'L4_to_L23', connOrder: CONN_ORDER }), 'pulse 1 of 4: L4 → L23');
  assert.equal(columnNarration({ conn: 'L6_to_L4', connOrder: CONN_ORDER }), 'pulse 4 of 4: L6 → L4');
});
