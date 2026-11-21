import { PredictiveCodingNetwork, PredictiveSynapseModel } from '../js/models/learning_numenta.js';

export const meta = {
  id: 'learn-predictive',
  title: 'Predictive coding learning',
  conex: ['PredictiveSynapseLearning', 'PredictiveCodingLearning', 'PrecisionWeighting'],
  source: 'conex/behaviors/neurons/numenta/predictive_coding.py',
};

const SIZE = 8;
const STEPS = 25;

function mulberry(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- Parameters section (CoNeX defaults) ---
  const sec = ui.section(controls, 'Learning & precision parameters');
  const lr = ui.slider(sec, {
    key: 'learningRate', label: 'learning_rate', unit: 'a.u.',
    min: 0.005, max: 0.1, step: 0.005, value: 0.05,
  });
  const precLr = ui.slider(sec, {
    key: 'precisionLearningRate', label: 'precision_learning_rate', unit: 'a.u.',
    min: 0.001, max: 0.05, step: 0.001, value: 0.01,
  });
  const precTau = ui.slider(sec, {
    key: 'precisionTau', label: 'precision_tau', unit: 'ms',
    min: 5, max: 100, step: 5, value: 20,
  });
  const minPrec = ui.slider(sec, {
    key: 'minPrecision', label: 'min_precision', unit: 'a.u.',
    min: 0.01, max: 1.0, step: 0.01, value: 0.01,
  });
  const maxPrec = ui.slider(sec, {
    key: 'maxPrecision', label: 'max_precision', unit: 'a.u.',
    min: 1, max: 100, step: 5, value: 100,
  });
  const nonlin = ui.select(sec, {
    key: 'errorNonlinearity', label: 'error_nonlinearity',
    options: [
      { value: 'linear', label: 'linear' },
      { value: 'relu', label: 'relu' },
      { value: 'tanh', label: 'tanh' },
    ],
    value: 'linear',
  });
  const useHebbian = ui.toggle(sec, {
    key: 'useHebbian', label: 'use_hebbian', value: false,
  });
  const lType = ui.select(sec, {
    key: 'learningType', label: 'synapse learning_type',
    options: [
      { value: 'error_driven', label: 'error_driven' },
      { value: 'hebbian', label: 'hebbian' },
      { value: 'combined', label: 'combined' },
    ],
    value: 'error_driven',
  });
  const seedCtrl = ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: 7 });

  // --- Figures ---
  const heatmapFig = ui.heatmap(figures, {
    rows: SIZE,
    cols: SIZE,
    label: 'Top-down prediction weights matrix (W)',
  });
  const statusNote = ui.note(figures, '');
  const errPlot = ui.plot(figures, {
    height: 200,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'mean |error| / free energy',
    yUnit: 'a.u.',
  });
  const precPlot = ui.plot(figures, {
    height: 180,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'precision',
    yUnit: 'a.u.',
  });

  let currentStep = 0;
  let trace = [];

  function runSimulation() {
    const rng = mulberry(seedCtrl.get());
    const net = new PredictiveCodingNetwork(SIZE, {
      learningRate: lr.get(),
      precisionTau: precTau.get(),
      precisionLearningRate: precLr.get(),
      minPrecision: minPrec.get(),
      maxPrecision: maxPrec.get(),
      errorNonlinearity: nonlin.get(),
      useHebbian: useHebbian.get(),
      useSparseErrors: false,
    });

    // Also run synapse model in parallel to verify eligibility traces
    const synModel = new PredictiveSynapseModel({
      srcSize: SIZE,
      dstSize: SIZE,
      learningRate: lr.get(),
      learningType: lType.get(),
      initialWeights: Array.from(net.topDownWeights),
    });

    const sensory = new Float64Array(SIZE);
    const higher = new Float64Array(SIZE);
    for (let i = 0; i < SIZE; i++) {
      sensory[i] = rng() * 0.8 + 0.1;
      higher[i] = rng() * 0.8 + 0.1;
    }
    net.activity = Float64Array.from(sensory);
    net.topDownInput = Float64Array.from(higher);

    trace = [];
    for (let t = 0; t < STEPS; t++) {
      const stepRes = net.step();
      synModel.step({
        srcActivity: net.topDownInput,
        dstError: net.predictionError,
        dstActivity: net.activity,
      });

      trace.push({
        step: t,
        meanAbsError: stepRes.meanAbsError,
        meanPrecision: stepRes.meanPrecision,
        freeEnergy: stepRes.freeEnergy,
        prediction: stepRes.prediction,
        signedError: stepRes.signedError,
        precision: stepRes.precision,
        weights: stepRes.weights,
        synWeights: synModel.weights,
      });
    }
  }

  function renderAt(step) {
    if (!trace || trace.length === 0) return;
    const clampedStep = Math.max(0, Math.min(step, STEPS - 1));
    currentStep = clampedStep;
    const s = trace[clampedStep];

    // 1. Draw weights heatmap
    let minW = Infinity;
    let maxW = -Infinity;
    for (let i = 0; i < s.weights.length; i++) {
      if (s.weights[i] < minW) minW = s.weights[i];
      if (s.weights[i] > maxW) maxW = s.weights[i];
    }
    if (minW === maxW) {
      minW -= 0.1;
      maxW += 0.1;
    }
    heatmapFig.draw(s.weights, { min: minW, max: maxW });

    // 2. Status readout
    statusNote.innerHTML = `
      <div style="font-family:ui-monospace,monospace;font-size:0.85rem;line-height:1.4;">
        <strong>Step ${clampedStep}</strong>: Mean |error| = <span style="color:var(--c2,#d95b43);font-weight:bold;">${s.meanAbsError.toFixed(4)}</span>
        &bull; Mean precision = <span style="color:var(--c1,#0f7b7f);font-weight:bold;">${s.meanPrecision.toFixed(2)}</span>
        &bull; Free energy = ${s.freeEnergy.toFixed(4)}
        &bull; Weight range = [${minW.toFixed(3)}, ${maxW.toFixed(3)}]
      </div>
    `;

    // 3. Error and free energy curves
    errPlot.draw(
      [
        {
          x: trace.map((p) => p.step),
          y: trace.map((p) => p.meanAbsError),
          label: 'mean |prediction error|',
          color: 'c2',
        },
        {
          x: trace.map((p) => p.step),
          y: trace.map((p) => p.freeEnergy),
          label: 'free energy F',
          color: 'c4',
        },
      ],
      {
        xRange: [0, STEPS - 1],
        revealX: clampedStep,
      },
    );

    // 4. Precision curve
    precPlot.draw(
      [
        {
          x: trace.map((p) => p.step),
          y: trace.map((p) => p.meanPrecision),
          label: 'mean precision',
          color: 'c1',
        },
      ],
      {
        xRange: [0, STEPS - 1],
        revealX: clampedStep,
        hlines: [
          { y: 1.0, label: 'baseline precision (1.0)' },
        ],
      },
    );
  }

  // --- Animation / Player ---
  let player = null;
  if (typeof ui.player === 'function') {
    player = ui.player(figures, {
      duration: STEPS - 1,
      unit: 'step',
      playSeconds: 6,
      loop: true,
      onFrame: (t) => {
        renderAt(Math.round(t));
      },
    });
  }

  function rebuild() {
    runSimulation();
    renderAt(currentStep);
  }

  [lr, precLr, precTau, minPrec, maxPrec, nonlin, useHebbian, lType, seedCtrl].forEach((c) => {
    c.onInput(rebuild);
  });

  ui.onTheme(() => {
    renderAt(currentStep);
  });

  rebuild();

  // --- Teaching notes ---
  ui.note(notes, `
    <p>Predictive coding models neocortical computation as hierarchical Bayesian inference (Rao &amp; Ballard, 1999; Friston, 2005).
    Higher cortical areas generate top-down predictions of sensory representations in lower areas. Lower areas compare these
    predictions against actual incoming drive, computing a residual <strong>prediction error</strong> that ascends the hierarchy.</p>
    <p>Learning in CoNeX updates the feedback prediction weights via gradient descent on prediction error
    (<code>PredictiveSynapseLearning</code> with eligibility traces, and <code>PredictiveCodingLearning</code>).
    As the synaptic weights learn the mapping from the higher-level pattern to the sensory activity, the prediction error falls monotonically.</p>
    <p><strong>Precision weighting</strong> (<code>PrecisionWeighting._learn_precision</code>) estimates the inverse variance of the error signal online.
    Reliable, low-noise units develop high precision, amplifying their prediction errors, whereas noisy or uncertain units are down-weighted.</p>
    <p><em>Note on CoNeX hierarchy:</em> In <code>conex/behaviors/neurons/numenta/predictive_coding.py</code> (around line 753),
    the multi-level <code>HierarchicalPredictiveCoding</code> class indexes top-down predictions using <code>level_predictions[i + 1]</code>
    for every level without binding the top level to the propagated feedforward input, leaving top-level predictions at zero.
    This plate uses the robust single-level predictive learning behaviors (<code>TopDownPrediction</code>, <code>ErrorUnit</code>,
    <code>PrecisionWeighting</code>, and <code>PredictiveCodingLearning</code>) that accurately minimize variational free energy.</p>
  `);

  ui.equation(notes, `
    <p>Prediction: <var>pred</var> = <var>W</var><sup>T</sup> &middot; <var>top_down_input</var></p>
    <p>Signed error: <var>e</var><sub>signed</sub> = <var>activity</var> &minus; <var>pred</var>, &nbsp; <var>e</var> = <var>f</var>(<var>e</var><sub>signed</sub>)</p>
    <p>Synapse update: &Delta;<var>W</var> = <var>top_down_input</var> &otimes; <var>e</var>, &nbsp; <var>eligibility</var> &larr; 0.9 &middot; <var>eligibility</var> + 0.1 &Delta;<var>W</var></p>
    <p>Precision gradient: &part;<var>F</var>/&part;<var>&gamma;</var> = 0.5 &middot; (<var>e</var><sub>signed</sub><sup>2</sup> &minus; 1 / (<var>&gamma;</var> + 10<sup>&minus;8</sup>))</p>
    <p>Variational free energy: <var>F</var> = 0.5 &sum; <var>&gamma;</var> &middot; <var>e</var><sub>signed</sub><sup>2</sup> &minus; 0.5 &sum; log(<var>&gamma;</var> + 10<sup>&minus;8</sup>)</p>
  `);

  ui.note(notes, `
    <p><strong>Constants &amp; Sources:</strong> Rao &amp; Ballard (1999) <em>Nature Neuroscience</em>;
    Friston (2005) <em>Philosophical Transactions of the Royal Society B</em>;
    defaults from <code>PredictiveCodingConfig</code> in CoNeX.</p>
    <p><strong>ML analogy:</strong> An autoencoder bottleneck performing online gradient descent on reconstruction error,
    where precision acts as a learnable per-feature loss weight.</p>
    <p><strong>Where the analogy breaks:</strong> Deep learning autoencoders require end-to-end backpropagation through a global scalar loss.
    In predictive coding, all updates are purely local (pre-synaptic drive multiplied by post-synaptic prediction error),
    using biologically plausible eligibility traces and local Bayesian surprise minimization.</p>
  `);

  // --- CoNeX code box ---
  ui.code(notes, () => `import torch
from pymonntorch import Network, NeuronGroup
from conex import (
    TimeResolution,
    PredictiveCodingConfig,
    PredictiveCodingLearning,
    PrecisionWeighting,
    ErrorUnit,
    TopDownPrediction,
    FreeEnergyMinimization,
)

config = PredictiveCodingConfig(
    learning_rate=${lr.get()},
    precision_tau=${precTau.get()},
    precision_learning_rate=${precLr.get()},
    min_precision=${minPrec.get()},
    max_precision=${maxPrec.get()},
    error_nonlinearity="${nonlin.get()}",
    use_sparse_errors=False,
)
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
behaviors = {
    230: TopDownPrediction(config=config),
    234: ErrorUnit(config=config, precision_weighted=False),
    236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
    238: FreeEnergyMinimization(config=config, track_components=True),
    350: PredictiveCodingLearning(config=config, use_hebbian=${useHebbian.get() ? 'True' : 'False'}),
}
neurons = NeuronGroup(net=net, size=${SIZE}, behavior=behaviors, tag="predictive_level")
net.initialize()
`);

  return {
    destroy() {
      if (player && typeof player.pause === 'function') player.pause();
    },
  };
}
