import { PredictiveCodingNetwork, PredictiveSynapseModel } from '../js/models/learning_numenta.js';

export const meta = {
  id: 'learn-predictive',
  title: 'Predictive coding learning',
  conex: ['PredictiveSynapseLearning', 'PredictiveCodingLearning', 'PrecisionWeighting'],
  source: 'conex/behaviors/neurons/numenta/predictive_coding.py',
};

const SIZE = 8;
const STEPS = 25;

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/learning.test.mjs can test them by hand-computed values.

// A weight cell's colour fraction (0..1) over the [min, max] range.
export function weightFraction(w, min, max) {
  if (max <= min) return 0;
  return Math.max(0, Math.min(1, (w - min) / (max - min)));
}

// An error particle's radius (px) shrinks as the mean |error| falls.
export function errorParticleRadius(meanAbsError, maxError, maxRadius) {
  if (maxError <= 0) return 0;
  return Math.max(0, Math.min(1, meanAbsError / maxError)) * maxRadius;
}

// One-line narration driven by the model state.
export function learnNarration({ step, meanAbsError, freeEnergy, meanPrecision }) {
  return `step ${step}: weights are learning the mapping (mean |error| ${meanAbsError.toFixed(4)} a.u., free energy ${freeEnergy.toFixed(4)} a.u., precision ${meanPrecision.toFixed(3)} a.u.)`;
}

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

  // --- Parameters section (CoNeX PredictiveCodingConfig & PredictiveCodingLearning defaults) ---
  const sec = ui.section(controls, 'Learning & precision parameters');
  const lr = ui.slider(sec, {
    key: 'learningRate', label: 'learning_rate', unit: 'a.u.',
    min: 0.001, max: 0.1, step: 0.001, value: 0.01,
  });
  const precLr = ui.slider(sec, {
    key: 'precisionLearningRate', label: 'precision_learning_rate', unit: 'a.u.',
    min: 0.0005, max: 0.05, step: 0.0005, value: 0.001,
  });
  const precTau = ui.slider(sec, {
    key: 'precisionTau', label: 'precision_tau', unit: 'ms',
    min: 5, max: 200, step: 5, value: 100,
  });
  const minPrec = ui.slider(sec, {
    key: 'minPrecision', label: 'min_precision', unit: 'a.u.',
    min: 0.005, max: 1.0, step: 0.005, value: 0.01,
  });
  const maxPrec = ui.slider(sec, {
    key: 'maxPrecision', label: 'max_precision', unit: 'a.u.',
    min: 1, max: 200, step: 5, value: 100,
  });
  const nonlin = ui.select(sec, {
    key: 'errorNonlinearity', label: 'error_nonlinearity',
    options: [
      { value: 'relu', label: 'relu' },
      { value: 'linear', label: 'linear' },
      { value: 'tanh', label: 'tanh' },
    ],
    value: 'relu',
  });
  const useHebbian = ui.toggle(sec, {
    key: 'useHebbian', label: 'use_hebbian', value: true,
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

  ui.tryThis(figures, 'Try this: raise <code>learning_rate</code> and watch the weight matrix settle and the error fall faster.');

  // --- Figures: hero (weight matrix + error particles) on top, plots below ---
  const hero = ui.hero(figures, { height: 300 });
  const colorBarNote = ui.note(figures, '');
  const statusNote = ui.note(figures, '');
  const errPlot = ui.plot(figures, {
    height: 200,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'mean |error|',
    yUnit: 'a.u.',
  });
  const fePlot = ui.plot(figures, {
    height: 180,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'free energy',
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
  let maxError = 1;

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
    maxError = 1;
    for (let t = 0; t < STEPS; t++) {
      const stepRes = net.step();
      synModel.step({
        srcActivity: net.topDownInput,
        dstError: net.predictionError,
        dstActivity: net.activity,
      });
      maxError = Math.max(maxError, stepRes.meanAbsError);
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

  // Hero: the weight matrix morphs (eased) between step k and k+1; the error
  // particle shrinks as the mean |error| falls.
  function drawHero(stepFloat) {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const i = Math.max(0, Math.min(STEPS - 1, Math.floor(stepFloat)));
    const j = Math.min(STEPS - 1, i + 1);
    const e = ui.ease(stepFloat - i); // ease-in-out between step k and k+1
    const s = trace[i];
    const sN = trace[j];
    if (!s) return;

    // weight matrix: square cells, eased between the two steps
    const n = SIZE;
    const side = Math.min(width - 120, height - 64);
    const ox = 40;
    const oy = 20;
    const cw = side / n;
    let minW = Infinity, maxW = -Infinity;
    for (let k = 0; k < n * n; k++) {
      const w = s.weights[k] + (sN.weights[k] - s.weights[k]) * e;
      if (w < minW) minW = w;
      if (w > maxW) maxW = w;
    }
    if (minW === maxW) { minW -= 0.1; maxW += 0.1; }
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const w = s.weights[r * n + c] + (sN.weights[r * n + c] - s.weights[r * n + c]) * e;
        const f = weightFraction(w, minW, maxW);
        ctx.fillStyle = mixHex(ui.color('paper'), c1, f);
        ctx.fillRect(ox + c * cw, oy + r * cw, cw + 0.5, cw + 0.5);
      }
    }
    ctx.strokeStyle = ink;
    ctx.strokeRect(ox, oy, side, side);
    // row / column indices
    ctx.fillStyle = muted;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let k = 0; k < n; k++) {
      ctx.fillText(String(k), ox + (k + 0.5) * cw, oy + side + 10); // target j
      ctx.textAlign = 'right';
      ctx.fillText(String(k), ox - 6, oy + (k + 0.5) * cw); // source i
      ctx.textAlign = 'center';
    }
    // Axis titles: "source i" above the rows, "target j" centred under the
    // columns. Kept off the right edge so they never collide with the error
    // particle (which sits to the right of the matrix).
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('source i →', ox, oy - 8);
    ctx.textAlign = 'center';
    ctx.fillText('target j', ui.fitText(ctx, 'target j', ox + side / 2, width, 'center'), oy + side + 24);

    // error particles: radius shrinks as the mean |error| falls
    const err = s.meanAbsError + (sN.meanAbsError - s.meanAbsError) * e;
    const r = errorParticleRadius(err, maxError, 16);
    const cx = ox + side + 50;
    const cy = oy + side / 2;
    ctx.fillStyle = c2;
    ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, r), 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink;
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('|error|', cx, cy + Math.max(1, r) + 16);
    ctx.textAlign = 'left';

    hero.narration(learnNarration({
      step: i, meanAbsError: err, freeEnergy: s.freeEnergy, meanPrecision: s.meanPrecision,
    }));
  }

  function renderAt(step) {
    if (!trace || trace.length === 0) return;
    const clampedStep = Math.max(0, Math.min(step, STEPS - 1));
    currentStep = clampedStep;
    const s = trace[Math.floor(clampedStep)];

    // colour scale for the weight matrix
    let minW = Infinity, maxW = -Infinity;
    for (let i = 0; i < s.weights.length; i++) {
      if (s.weights[i] < minW) minW = s.weights[i];
      if (s.weights[i] > maxW) maxW = s.weights[i];
    }
    if (minW === maxW) { minW -= 0.1; maxW += 0.1; }
    colorBarNote.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;max-width:340px;margin:2px auto;font-family:ui-monospace,monospace;font-size:0.75rem;">
        <span>min: <strong>${minW.toFixed(3)}</strong></span>
        <div style="flex:1;height:12px;margin:0 10px;border:1px solid var(--rule,#dcdccd);background:linear-gradient(to right, var(--paper,#fbfaf5), var(--c1,#0f7b7f));border-radius:2px;"></div>
        <span>max: <strong>${maxW.toFixed(3)}</strong></span>
      </div>
      <div style="text-align:center;font-size:0.75rem;color:var(--muted,#5f6b6d);margin-top:2px;margin-bottom:6px;">
        Weight matrix <strong>W</strong>: maps Level 1 (top-down source, rows 0&ndash;${SIZE - 1}) &rarr; Level 0 (sensory prediction, cols 0&ndash;${SIZE - 1}). Prediction: <em>pred = W<sup>T</sup> &middot; x<sub>top_down</sub></em>.
      </div>
    `;

    statusNote.innerHTML = `
      <div style="font-family:ui-monospace,monospace;font-size:0.85rem;line-height:1.4;">
        <strong>Step ${Math.floor(clampedStep)}</strong>: Mean |error| = <span style="color:var(--c2,#d95b43);font-weight:bold;">${s.meanAbsError.toFixed(4)}</span>
        &bull; Mean precision = <span style="color:var(--c1,#0f7b7f);font-weight:bold;">${s.meanPrecision.toFixed(3)}</span>
        &bull; Free energy = ${s.freeEnergy.toFixed(4)}
      </div>
    `;

    drawHero(clampedStep);

    // error and free energy on separate, named plots
    errPlot.draw(
      [
        {
          x: trace.map((p) => p.step),
          y: trace.map((p) => p.meanAbsError),
          label: 'mean |prediction error|',
          color: 'c2',
        },
      ],
      { xRange: [0, STEPS - 1], revealX: clampedStep },
    );
    fePlot.draw(
      [
        {
          x: trace.map((p) => p.step),
          y: trace.map((p) => p.freeEnergy),
          label: 'free energy F',
          color: 'c4',
        },
      ],
      { xRange: [0, STEPS - 1], revealX: clampedStep },
    );

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
        hlines: [{ y: 1.0, label: 'baseline precision (1.0)' }],
      },
    );
  }

  // --- Animation / Player (eased morph between steps) ---
  // A continuous unit so onFrame gets a fractional step index; the hero
  // narration names the current step.
  const player = ui.player(figures, {
    duration: STEPS - 1,
    unit: 'a.u.',
    playSeconds: 6,
    loop: true,
    onFrame: (t) => {
      renderAt(t);
    },
  });

  function rebuild() {
    runSimulation();
    renderAt(currentStep);
  }

  [lr, precLr, precTau, minPrec, maxPrec, nonlin, useHebbian, lType, seedCtrl].forEach((c) => {
    c.onInput(rebuild);
  });

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(() => { if (trace.length) drawHero(currentStep); });
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
    The hero shows the weight matrix <strong>W</strong> morphing, eased, from one learning step to the next; as the weights
    learn the mapping from the higher-level pattern to the sensory activity, the prediction error falls and the error
    particle (right) shrinks.</p>
    <p><strong>Precision weighting</strong> (<code>PrecisionWeighting._learn_precision</code>) estimates the inverse variance of the error signal online.
    Reliable, low-noise units develop high precision, amplifying their prediction errors, whereas noisy or uncertain units are down-weighted.</p>
    <p><em>Note on CoNeX hierarchy:</em> In <code>conex/behaviors/neurons/numenta/predictive_coding.py</code> (around line 753),
    the multi-level <code>HierarchicalPredictiveCoding</code> class indexes top-down predictions using <code>level_predictions[i + 1]</code>
    for every level without binding the top level to the propagated feedforward input, leaving top-level predictions at zero.
    This plate uses the single-level and synapse learning chain (<code>TopDownPrediction</code>, <code>ErrorUnit</code>,
    <code>PrecisionWeighting</code>, and <code>PredictiveCodingLearning</code>) which accurately executes online gradient descent.</p>
  `);

  ui.equation(notes, `
    <p>Prediction: <var>pred</var> = <var>W</var><sup>T</sup> &middot; <var>top_down_input</var></p>
    <p>Signed error: <var>e</var><sub>signed</sub> = <var>activity</var> &minus; <var>pred</var>, &nbsp; <var>e</var> = <var>f</var>(<var>e</var><sub>signed</sub>)</p>
    <p>Synapse update: &Delta;<var>W</var> = <var>top_down_input</var> &otimes; <var>e</var>, &nbsp; <var>eligibility</var> &larr; 0.9 &middot; <var>eligibility</var> + 0.1 &Delta;<var>W</var></p>
    <p>Precision gradient: &part;<var>F</var>/&part;<var>&gamma</var> = 0.5 &middot; (<var>e</var><sub>signed</sub><sup>2</sup> &minus; 1 / (<var>&gamma</var> + 10<sup>&minus;8</sup>))</p>
    <p>Variational free energy: <var>F</var> = 0.5 &sum; <var>&gamma</var> &middot; <var>e</var><sub>signed</sub><sup>2</sup> &minus; 0.5 &sum; log(<var>&gamma</var> + 10<sup>&minus;8</sup>)</p>
  `);

  ui.note(notes, `
    <p><strong>Constants &amp; Sources:</strong> Rao &amp; Ballard (1999) <em>Nature Neuroscience</em>;
    Friston (2005) <em>Philosophical Transactions of the Royal Society B</em>;
    defaults from <code>PredictiveCodingConfig</code> (<code>predictive_coding.py:45</code>) and <code>PredictiveCodingLearning</code> (<code>predictive_coding.py:490</code>) in CoNeX.</p>
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

// Blend two hex colors; t in [0,1].
function mixHex(a, b, t) {
  const pa = hex(a), pb = hex(b);
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function hex(s) {
  const m = s.match(/^#([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
