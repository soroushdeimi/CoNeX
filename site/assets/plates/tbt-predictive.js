import { PredictiveCodingLevel } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-predictive',
  title: 'Hierarchical predictive coding',
  conex: ['HierarchicalPredictiveCoding', 'PredictionUnit', 'ErrorUnit', 'PrecisionWeighting'],
  source: 'conex/behaviors/neurons/numenta/predictive_coding.py',
};

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// Particle spawn rate (particles / s) is proportional to the real magnitude.
export function particleRate(magnitude, maxMagnitude, maxRate) {
  if (maxMagnitude <= 0) return 0;
  return Math.max(0, Math.min(1, magnitude / maxMagnitude)) * maxRate;
}

// The precision gain knob position (0..1) over [minPrecision, maxPrecision].
export function precisionKnob(precision, minPrec, maxPrec) {
  if (maxPrec <= minPrec) return 0;
  return Math.max(0, Math.min(1, (precision - minPrec) / (maxPrec - minPrec)));
}

// One-line narration driven by the model state.
export function predictiveNarration({ meanAbs, freeEnergy, meanPrec }) {
  const trend = meanAbs < 0.05 ? 'error is small: the prediction fits the input' : 'error is large: the prediction is surprised';
  return `${trend} (mean |error| ${meanAbs.toFixed(3)} a.u., free energy ${freeEnergy.toFixed(3)} a.u., precision ${meanPrec.toFixed(2)} a.u.)`;
}

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
  // Re-running on every control change keeps the plots populated (G1).
  const rebuild = () => { build(); runSteps(40); currentStep = trace.length - 1; };
  [signalType, sigSeed, hiSeed, lr, precTau, precLr, minPrec, maxPrec, nonlin, sparse, errThresh].forEach((c) => c.onInput(rebuild));

  // --- figures: hero on top, plots below ---
  const hero = ui.hero(figures, { height: 260 });
  const errPlot = ui.plot(figures, { height: 220, xLabel: 'step', xUnit: 'index', yLabel: 'mean |error| / prediction', yUnit: 'a.u.' });
  const fePlot = ui.plot(figures, { height: 200, xLabel: 'step', xUnit: 'index', yLabel: 'free energy', yUnit: 'a.u.' });
  const precPlot = ui.plot(figures, { height: 200, xLabel: 'step', xUnit: 'index', yLabel: 'mean precision', yUnit: 'a.u.' });
  let currentStep = 0;

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

  // Hero: two level boxes. Teal particles flow down (prediction), coral
  // particles flow up (error); spawn rate is proportional to the real mean
  // prediction / mean |error| at the current step. Precision is a gain knob.
  let downParticles = [];
  let upParticles = [];
  const prng = mulberry(1234); // seeded jitter for particle spawn (decorative)
  function drawHero(p, dtMs) {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c5 = ui.color('c5');
    const cx = width / 2;
    const topY = 30, botY = height - 30;
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillStyle = ink;
    ctx.textAlign = 'center';
    ctx.fillText('level 1 (higher)', cx, topY - 10);
    ctx.fillText('level 0 (sensory)', cx, botY + 18);
    ctx.strokeStyle = muted;
    ctx.strokeRect(cx - 80, topY - 4, 160, 22);
    ctx.strokeRect(cx - 80, botY - 18, 160, 22);

    const err = p ? p.meanAbs : 0;
    const pred = p ? p.meanPred : 0;
    const prec = p ? p.meanPrec : 1;
    const dt = dtMs / 1000;

    // spawn particles at a rate proportional to the real magnitude
    const downRate = particleRate(pred, 1, 14); // particles / s
    const upRate = particleRate(err, 1, 14);
    const spawn = (arr, rate, x, y, vy) => {
      let n = Math.floor(rate * dt);
      if (prng.next() < rate * dt - n) n++; // fractional remainder
      for (let i = 0; i < n; i++) {
        arr.push({ x: x + (prng.next() - 0.5) * 10, y, vy, life: 1 });
      }
    };
    spawn(downParticles, downRate, cx + 26, topY + 18, 90);
    spawn(upParticles, upRate, cx - 26, botY - 18, -90);

    // prediction down (teal)
    ctx.fillStyle = c1;
    for (const pt of downParticles) {
      pt.y += pt.vy * dt;
      if (pt.y > botY - 18) pt.life = 0;
      ctx.globalAlpha = 0.8 * pt.life;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI); ctx.fill();
    }
    // error up (coral)
    ctx.fillStyle = c2;
    for (const pt of upParticles) {
      pt.y += pt.vy * dt;
      if (pt.y < topY + 18) pt.life = 0;
      ctx.globalAlpha = 0.8 * pt.life;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    downParticles = downParticles.filter((pt) => pt.life > 0);
    upParticles = upParticles.filter((pt) => pt.life > 0);

    // arrowheads
    arrowHead(ctx, cx - 26, topY + 18, c2, -1);
    arrowHead(ctx, cx + 26, botY - 18, c1, 1);
    ctx.textAlign = 'left';
    ctx.fillStyle = c2;
    ctx.fillText('error up', cx - 26 - 58, (topY + botY) / 2 - 6);
    ctx.fillStyle = c1;
    ctx.fillText('prediction down', cx + 34, (topY + botY) / 2 + 6);
    ctx.textAlign = 'center';

    // precision gain knob (right side)
    const kx = width - 46;
    const ky0 = topY + 18, ky1 = botY - 18;
    const knob = precisionKnob(prec, minPrec.get(), maxPrec.get());
    ctx.strokeStyle = muted;
    ctx.beginPath(); ctx.moveTo(kx, ky0); ctx.lineTo(kx, ky1); ctx.stroke();
    ctx.fillStyle = c5;
    const ky = ky1 - knob * (ky1 - ky0);
    ctx.beginPath(); ctx.arc(kx, ky, 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = ink;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('precision', kx, ky0 - 8);
    ctx.fillText(prec.toFixed(2), kx, ky1 + 14);

    hero.narration(predictiveNarration({ meanAbs: err, freeEnergy: p ? p.freeEnergy : 0, meanPrec: prec }));
  }
  function arrowHead(ctx, x, y, color, dir) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - 5, y + 8 * dir);
    ctx.lineTo(x + 5, y + 8 * dir);
    ctx.closePath();
    ctx.fill();
  }

  function render() {
    const t = trace;
    const revealX = t[currentStep] ? t[currentStep].step : 0;
    errPlot.draw(
      [
        { x: t.map((p) => p.step), y: t.map((p) => p.meanAbs), label: 'mean |error|', color: 'c2' },
        { x: t.map((p) => p.step), y: t.map((p) => p.meanPred), label: 'mean prediction', color: 'c1' },
      ],
      { xRange: [0, Math.max(1, t.length - 1)], revealX },
    );
    fePlot.draw(
      [{ x: t.map((p) => p.step), y: t.map((p) => p.freeEnergy), label: 'free energy', color: 'c4' }],
      { xRange: [0, Math.max(1, t.length - 1)], revealX },
    );
    precPlot.draw(
      [{ x: t.map((p) => p.step), y: t.map((p) => p.meanPrec), label: 'mean precision', color: 'c5' }],
      { xRange: [0, Math.max(1, t.length - 1)], revealX },
    );
    drawHero(t[currentStep], 16);
  }

  function onFrame(t) {
    const i = Math.max(0, Math.min(trace.length - 1, Math.round((t / 100) * (trace.length - 1))));
    currentStep = i;
    render();
  }

  ui.button(controls, { label: 'Run 40 steps', onClick: () => runSteps(40) });
  ui.button(controls, { label: 'Perturb input', onClick: perturb });
  ui.button(controls, { label: 'Reset', onClick: rebuild });

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(() => { if (trace[currentStep]) drawHero(trace[currentStep], 16); });

  build();
  // Run a default learning run on load so the plots are not empty; the buttons
  // only re-run, perturb, or reset.
  runSteps(40);
  currentStep = trace.length - 1; // start at the end so the plots are full
  render();
  ui.onTheme(render);
  ui.player(figures, {
    duration: 100,
    unit: 'a.u.',
    playSeconds: 6,
    loop: true,
    onFrame,
  });

  // --- notes ---
  ui.note(notes, `
    <p>Predictive coding is a hierarchy in which each level <em>predicts</em> the
    level below it and sends only the <strong>prediction error</strong> upward.
    Here level 1 (higher) provides a top-down pattern and level 0 (sensory) learns
    to predict its input from it. Each step runs the CoNeX chain
    <code>TopDownPrediction &rarr; ErrorUnit &rarr; PrecisionWeighting &rarr;
    FreeEnergyMinimization &rarr; PredictiveCodingLearning</code>.</p>
    <p>The hero shows the two levels. Teal particles flow <em>down</em> (the
    prediction) and coral particles flow <em>up</em> (the error); the rate at which
    each stream spawns is proportional to that step's real mean prediction and mean
    |error|. The knob on the right is the mean precision, the gain that weights how
    much each unit's error matters. As the top-down weights learn, the error stream
    thins and the free-energy plot (below) falls.</p>
    <p><strong>Perturb input</strong> injects noise into the sensory signal and the
    error jumps again &mdash; the "surprise" a real hierarchy would propagate upward.</p>
    <p><strong>Precision</strong> is the inverse of the error variance, learned
    online and clamped to [<code>min_precision</code>, <code>max_precision</code>].
    Note the free-energy quirk in CoNeX: with the error near zero the log-normaliser
    term dominates and precision keeps climbing, so the free energy keeps falling
    &mdash; there is no prior over the precision to give it a resting point.</p>
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
import torch
from pymonntorch import Network, NeuronGroup
from conex import (
    TopDownPrediction, ErrorUnit, PrecisionWeighting,
    FreeEnergyMinimization, PredictiveCodingLearning, PredictiveCodingConfig,
    TimeResolution,
)

config = PredictiveCodingConfig(
    learning_rate=${fmt(lr.get())},
    precision_tau=${fmt(precTau.get())},
    precision_learning_rate=${fmt(precLr.get())},
    min_precision=${fmt(minPrec.get())},
    max_precision=${fmt(maxPrec.get())},
    error_nonlinearity="${nonlin.get()}",
    use_sparse_errors=${sparse.get() ? 'True' : 'False'},
    error_threshold=${fmt(errThresh.get())},
)
behaviors = {
    230: TopDownPrediction(config=config),
    234: ErrorUnit(config=config, precision_weighted=False),
    236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
    238: FreeEnergyMinimization(config=config, track_components=True),
    350: PredictiveCodingLearning(config=config, use_hebbian=False),
}
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(net=net, size=${SIZE}, behavior=behaviors, tag="level")
net.initialize()

torch.manual_seed(${sigSeed.get()})
sensory = torch.rand(${SIZE})            # level 0 input
higher_level = torch.rand(${SIZE})       # level 1 pattern
neurons.activity = sensory
neurons.top_down_input = higher_level
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

// At most 4 significant digits for numbers shown in text and code snippets.
function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  return Number(v.toPrecision(4)).toString();
}
