import { PredictiveCodingLevel } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-predictive',
  title: 'Hierarchical predictive coding',
  conex: ['HierarchicalPredictiveCoding', 'PredictionUnit', 'ErrorUnit', 'PrecisionWeighting'],
  source: 'conex/behaviors/neurons/numenta/predictive_coding.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const SIZE = 32;

  // --- input signal (level 0, sensory) ---
  const sig = ui.section(controls, 'Sensory input (level 0)');
  const signalType = ui.select(sig, {
    key: 'signalType', label: 'signal',
    options: [
      { value: 'random', label: 'random' },
      { value: 'square', label: 'square wave' },
      { value: 'sine', label: 'sine wave' },
    ],
    value: 'random',
  });
  const sigSeed = ui.number(sig, { key: 'sigSeed', label: 'signal seed', unit: '', value: 7 });

  // --- higher level (level 1) top-down input ---
  const hi = ui.section(controls, 'Higher level (level 1)');
  const hiSeed = ui.number(hi, { key: 'hiSeed', label: 'higher-level seed', unit: '', value: 7 });

  // --- PredictiveCodingConfig parameters (CoNeX names and defaults) ---
  const cfg = ui.section(controls, 'PredictiveCodingConfig');
  const lr = ui.slider(cfg, { key: 'learningRate', label: 'learning_rate', unit: 'a.u.', min: 0.01, max: 0.1, step: 0.005, value: 0.05 });
  const precTau = ui.slider(cfg, { key: 'precisionTau', label: 'precision_tau', unit: 'ms', min: 5, max: 100, step: 5, value: 20 });
  const precLr = ui.slider(cfg, { key: 'precisionLearningRate', label: 'precision_learning_rate', unit: 'a.u.', min: 0.001, max: 0.05, step: 0.001, value: 0.01 });
  const minPrec = ui.slider(cfg, { key: 'minPrecision', label: 'min_precision', unit: 'a.u.', min: 0.01, max: 1, step: 0.01, value: 0.01 });
  const maxPrec = ui.slider(cfg, { key: 'maxPrecision', label: 'max_precision', unit: 'a.u.', min: 1, max: 100, step: 1, value: 100 });
  const nonlin = ui.select(cfg, {
    key: 'errorNonlinearity', label: 'error_nonlinearity',
    options: [
      { value: 'linear', label: 'linear' },
      { value: 'relu', label: 'relu' },
      { value: 'tanh', label: 'tanh' },
    ],
    value: 'linear',
  });
  const sparse = ui.toggle(cfg, { key: 'useSparseErrors', label: 'use_sparse_errors', value: false });
  const errThresh = ui.slider(cfg, { key: 'errorThreshold', label: 'error_threshold', unit: 'a.u.', min: 0, max: 0.5, step: 0.05, value: 0.1 });

  // --- state ---
  let level = null;
  let sensory = null;
  let higher = null;
  const trace = [];

  function makeSignal(type, seed) {
    const rng = mulberry(seed);
    const v = new Float64Array(SIZE);
    if (type === 'square') {
      for (let i = 0; i < SIZE; i++) v[i] = i < SIZE / 2 ? 1 : 0;
    } else if (type === 'sine') {
      for (let i = 0; i < SIZE; i++) v[i] = (Math.sin(2 * Math.PI * i / SIZE) + 1) / 2;
    } else {
      for (let i = 0; i < SIZE; i++) v[i] = rng();
    }
    return v;
  }

  function build() {
    level = new PredictiveCodingLevel(SIZE, {
      learningRate: lr.get(),
      precisionTau: precTau.get(),
      precisionLearningRate: precLr.get(),
      minPrecision: minPrec.get(),
      maxPrecision: maxPrec.get(),
      errorNonlinearity: nonlin.get(),
      useSparseErrors: sparse.get(),
      errorThreshold: errThresh.get(),
    });
    sensory = makeSignal(signalType.get(), sigSeed.get());
    higher = makeSignal('random', hiSeed.get());
    level.activity = Float64Array.from(sensory);
    level.topDownInput = Float64Array.from(higher);
    trace.length = 0;
  }
  const rebuild = () => { build(); render(); };
  [signalType, sigSeed, hiSeed, lr, precTau, precLr, minPrec, maxPrec, nonlin, sparse, errThresh].forEach((c) => c.onInput(rebuild));

  // --- figures ---
  const errPlot = ui.plot(figures, { height: 220, xLabel: 'step', xUnit: '', yLabel: 'mean |error| / prediction', yUnit: 'a.u.' });
  const fePlot = ui.plot(figures, { height: 200, xLabel: 'step', xUnit: '', yLabel: 'free energy', yUnit: 'a.u.' });
  const precPlot = ui.plot(figures, { height: 200, xLabel: 'step', xUnit: '', yLabel: 'mean precision', yUnit: 'a.u.' });

  function record() {
    const meanAbs = level.errorMagnitude.reduce((a, b) => a + b, 0) / SIZE;
    const meanPred = level.prediction.reduce((a, b) => a + b, 0) / SIZE;
    const meanPrec = level.precision.reduce((a, b) => a + b, 0) / SIZE;
    trace.push({
      step: trace.length,
      meanAbs,
      meanPred,
      freeEnergy: level.freeEnergy,
      meanPrec,
    });
    if (trace.length > 400) trace.shift();
  }

  function runSteps(n) {
    for (let i = 0; i < n; i++) {
      level.forward();
      record();
    }
    render();
  }

  function perturb() {
    const rng = mulberry(sigSeed.get() + 9999 + trace.length);
    for (let i = 0; i < SIZE; i++) level.activity[i] += (rng() - 0.5) * 0.8;
    runSteps(6);
  }

  function render() {
    const t = trace;
    errPlot.draw(
      [
        { x: t.map((p) => p.step), y: t.map((p) => p.meanAbs), label: 'mean |error|', color: 'c2' },
        { x: t.map((p) => p.step), y: t.map((p) => p.meanPred), label: 'mean prediction', color: 'c1' },
      ],
      { xRange: [0, Math.max(1, t.length - 1)] },
    );
    fePlot.draw(
      [{ x: t.map((p) => p.step), y: t.map((p) => p.freeEnergy), label: 'free energy', color: 'c4' }],
      { xRange: [0, Math.max(1, t.length - 1)] },
    );
    precPlot.draw(
      [{ x: t.map((p) => p.step), y: t.map((p) => p.meanPrec), label: 'mean precision', color: 'c5' }],
      { xRange: [0, Math.max(1, t.length - 1)] },
    );
  }

  ui.button(controls, { label: 'Run 40 steps', onClick: () => runSteps(40) });
  ui.button(controls, { label: 'Perturb input', onClick: perturb });
  ui.button(controls, { label: 'Reset', onClick: rebuild });

  build();
  render();
  ui.onTheme(render);

  // --- notes ---
  ui.note(notes, `
    <p>Predictive coding is a hierarchy in which each level <em>predicts</em> the
    level below it and sends only the <strong>prediction error</strong> upward.
    Here level 1 (higher) provides a top-down pattern and level 0 (sensory) learns
    to predict its input from it. Each step runs the CoNeX chain
    <code>TopDownPrediction &rarr; ErrorUnit &rarr; PrecisionWeighting &rarr;
    FreeEnergyMinimization &rarr; PredictiveCodingLearning</code>.</p>
    <p>As the top-down weights learn the mapping, the mean |error| and the free
    energy both fall. <strong>Perturb input</strong> injects noise into the sensory
    signal and the error jumps again &mdash; the "surprise" a real hierarchy would
    propagate upward.</p>
    <p><strong>Precision</strong> is the inverse of the error variance, learned
    online and clamped to [<code>min_precision</code>, <code>max_precision</code>].
    It weights how much each unit's error matters. Note the free-energy quirk in
    CoNeX: with the error near zero the log-normaliser term dominates and precision
    keeps climbing, so the free energy keeps falling &mdash; there is no prior over
    the precision to give it a resting point.</p>
    <p>A note on scope: this plate runs CoNeX's single-level predictive-coding
    chain &mdash; the wiring of
    <code>Example/numenta/predictive_coding.py</code>. CoNeX also has
    <code>HierarchicalPredictiveCoding</code>, but in the current code its top-down
    pass reads <code>level_predictions[i + 1]</code> for every level
    (<code>conex/behaviors/neurons/numenta/predictive_coding.py</code>, around line
    753), so the top level never receives the propagated input and its predictions
    stay zero. For that reason the multi-level class is not animated here.</p>
  `);
  ui.equation(notes, `
    <p>prediction = W<sub>top</sub><sup>T</sup> &middot; top_down_input</p>
    <p>signed_error = activity &minus; prediction, &nbsp; error = nonlinearity(signed_error)</p>
    <p>precision = clamp(1 / (EMA(signed_error&sup2;) + 1e-8), min, max)</p>
    <p>free energy = 0.5 &sum; precision &middot; error&sup2; &minus; 0.5 &sum; log(precision)</p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> a predictive coding level is like a
    reconstruction network (an autoencoder bottleneck) that minimises the
    reconstruction error, with precision playing the role of a per-unit loss
    weight. The analogy breaks because the "weights" are updated by a local,
    error-driven rule with online precision estimation rather than backpropagation
    through a global loss, and the free energy is a Gaussian surprise bound, not a
    cross-entropy.</p>
  `);

  ui.code(notes, () => `
from conex import (
    TopDownPrediction, ErrorUnit, PrecisionWeighting,
    FreeEnergyMinimization, PredictiveCodingLearning, PredictiveCodingConfig,
)

config = PredictiveCodingConfig(
    learning_rate=${lr.get()},
    precision_tau=${precTau.get()},
    precision_learning_rate=${precLr.get()},
    min_precision=${minPrec.get()},
    max_precision=${maxPrec.get()},
    error_nonlinearity="${nonlin.get()}",
    use_sparse_errors=${sparse.get() ? 'True' : 'False'},
    error_threshold=${errThresh.get()},
)
behaviors = {
    230: TopDownPrediction(config=config),
    234: ErrorUnit(config=config, precision_weighted=False),
    236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
    238: FreeEnergyMinimization(config=config, track_components=True),
    350: PredictiveCodingLearning(config=config, use_hebbian=False),
}
neurons.activity = sensory            # level 0 input
neurons.top_down_input = higher_level  # level 1 pattern
for behavior in behaviors.values():
    behavior.forward(neurons)
`);

  return { destroy() {} };
}

function mulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
