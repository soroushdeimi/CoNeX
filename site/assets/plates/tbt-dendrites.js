import { DendriticSegment, ContextualPrediction } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-dendrites',
  title: 'Active dendrites',
  conex: ['DendriticSegment', 'ActiveDendriteComputation', 'ContextualPrediction'],
  source: 'conex/behaviors/neurons/numenta/active_dendrites.py',
};

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// A segment gauge's fill (0..1) is its activation over the activation threshold.
export function gaugeFill(activation, threshold) {
  if (threshold <= 0) return 0;
  return Math.max(0, Math.min(1, activation / threshold));
}

// Which colour a segment gauge uses: active, matching, or below.
export function segmentState(activation, actThresh, learnThresh) {
  if (activation >= actThresh) return 'active';
  if (activation >= learnThresh) return 'matching';
  return 'below';
}

// A cell is predictive when its depolarization crosses the threshold.
export function isPredictive(depol, depolThresh) {
  return depol >= depolThresh;
}

// One-line narration driven by the model state.
export function dendriteNarration({ k, nActive, nPredictive }) {
  if (k === 0) return 'no context yet: all segments are silent';
  return `${k} of ${nActive} context bits arrived; ${nPredictive} cell${nPredictive === 1 ? '' : 's'} now predictive`;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- DendriticSegment parameters (CoNeX names and defaults) ---
  const sec = ui.section(controls, 'DendriticSegment');
  const nCells = ui.number(sec, { key: 'nCells', label: 'n_cells', unit: 'cells', value: 12 });
  const nSegments = ui.number(sec, { key: 'nSegments', label: 'n_segments', unit: 'per cell', value: 4 });
  const nSyn = ui.number(sec, { key: 'nSynapses', label: 'n_synapses_per_segment', unit: 'syn', value: 6 });
  const actThresh = ui.slider(sec, {
    key: 'activationThreshold', label: 'activation_threshold', unit: 'syn',
    min: 1, max: 12, step: 1, value: 6,
  });
  const learnThresh = ui.slider(sec, {
    key: 'learningThreshold', label: 'learning_threshold', unit: 'syn',
    min: 1, max: 12, step: 1, value: 4,
  });
  const connPerm = ui.slider(sec, {
    key: 'connectedPermanence', label: 'connected_permanence', unit: 'a.u.',
    min: 0, max: 1, step: 0.05, value: 0.2,
  });

  // --- context SDR ---
  const ctx = ui.section(controls, 'Context SDR');
  const ctxSparsity = ui.slider(ctx, {
    key: 'ctxSparsity', label: 'context sparsity', unit: 'fraction',
    min: 0.05, max: 1, step: 0.05, value: 0.5,
  });
  const seed = ui.number(ctx, { key: 'seed', label: 'seed', unit: '', value: 7 });
  const depolThresh = ui.slider(ctx, {
    key: 'depolThreshold', label: 'depolarization_threshold', unit: 'a.u.',
    min: 0.1, max: 1, step: 0.05, value: 0.5,
  });

  let model = buildModel();
  function buildModel() {
    const nc = nCells.get();
    const ns = nSegments.get();
    const nS = nSyn.get();
    const seg = new DendriticSegment({
      nCells: nc, nSegments: ns, nSynapsesPerSegment: nS,
      activationThreshold: actThresh.get(),
      learningThreshold: learnThresh.get(),
      connectedPermanence: connPerm.get(),
    });
    // Deterministic wiring: each segment points at a seeded set of presynaptic cells.
    const rng = mulberry(seed.get());
    for (let s = 0; s < nc * ns; s++) {
      const targets = pickDistinct(rng, nc, nS);
      for (let j = 0; j < nS; j++) {
        seg.segmentTargets[s * nS + j] = targets[j];
        seg.segmentPermanences[s * nS + j] = 0.9; // connected
      }
    }
    const context = new ContextualPrediction({
      nCells: nc,
      depolarizationThreshold: depolThresh.get(),
      predictionWindow: 10,
      burstThreshold: 0.3,
    });
    return { seg, context, nc, ns, nS };
  }

  // --- figures: hero on top, depolarization plot below ---
  const hero = ui.hero(figures, { height: 300 });
  const depolPlot = ui.plot(figures, {
    height: 200, xLabel: 'cell', xUnit: 'index', yLabel: 'depolarization', yUnit: 'a.u.',
  });

  // The context SDR is revealed bit by bit. Precompute one frame per bit so the
  // animation only indexes, never recomputes.
  let frames = [];
  let order = [];
  function precompute() {
    const { seg, context, nc } = model;
    const rng = mulberry(seed.get() + 1000);
    const nActive = Math.max(1, Math.round(nc * ctxSparsity.get()));
    order = pickDistinct(rng, nc, nActive);
    frames = [];
    const active = new Array(nc).fill(false);
    for (let k = 0; k <= nActive; k++) {
      if (k > 0) active[order[k - 1]] = true;
      seg.computeSegmentActivation(active);
      seg.computeDepolarization();
      context.updatePredictions(null, seg.depolarization);
      frames.push({
        active: active.slice(),
        activation: Float64Array.from(seg.segmentActivation),
        depol: Float64Array.from(seg.depolarization),
        predictive: Uint8Array.from(context.predictive),
      });
    }
  }

  let currentFrame = null;
  let currentK = 0;

  function drawHero() {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const { seg, nc, ns } = model;
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c3 = ui.color('c3');
    const f = currentFrame;
    const actArr = f ? f.activation : seg.segmentActivation;
    const predArr = f ? f.predictive : null;
    const activeArr = f ? f.active : new Array(nc).fill(false);

    const cellW = width / nc;
    const topY = 22; // context row
    const gridTop = 48;
    const rowH = (height - gridTop - 34) / ns;

    // context row: one dot per presynaptic cell, lit when active
    for (let c = 0; c < nc; c++) {
      const cx = c * cellW + cellW / 2;
      ctx.fillStyle = activeArr[c] ? c1 : muted;
      ctx.globalAlpha = activeArr[c] ? 1 : 0.4;
      ctx.beginPath(); ctx.arc(cx, topY, 4, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = muted;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('context', 4, topY - 8);

    // segment gauges: one bar per (cell, segment), fill = activation/threshold
    for (let c = 0; c < nc; c++) {
      for (let s = 0; s < ns; s++) {
        const idx = c * ns + s;
        const act = actArr[idx];
        const x = c * cellW + 3;
        const y = gridTop + s * rowH;
        const w = cellW - 6;
        const h = rowH - 4;
        const frac = gaugeFill(act, actThresh.get());
        const state = segmentState(act, actThresh.get(), learnThresh.get());
        ctx.globalAlpha = 0.25 + 0.75 * frac;
        ctx.fillStyle = state === 'active' ? c1 : state === 'matching' ? c3 : muted;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = muted;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = ink;
        ctx.font = '11px ui-monospace, monospace';
        ctx.fillText(String(act), x + w / 2 - 4, y + h / 2 + 4);
      }
      // cell index, with a predictive glow when depolarization crosses the threshold
      ctx.fillStyle = muted;
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(String(c), c * cellW + cellW / 2 - 3, gridTop - 6);
      if (predArr && predArr[c]) {
        ctx.fillStyle = c1;
        ctx.beginPath();
        ctx.arc(c * cellW + cellW / 2 + 8, gridTop - 9, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // particles: the arriving context bit travels down to the segments that
    // synapse onto it (decorative; does not change any shown number)
    const b = currentK; // 0..nActive, fractional
    const k = Math.floor(b);
    const frac = b - k;
    if (k < order.length && frac > 0 && frac < 1) {
      const src = order[k]; // the presynaptic cell arriving now
      const srcX = src * cellW + cellW / 2;
      const nP = 3; // particles per arriving bit
      for (let s = 0; s < ns; s++) {
        for (let c = 0; c < nc; c++) {
          const segIdx = c * ns + s;
          let hits = 0;
          for (let j = 0; j < model.nS; j++) {
            if (seg.segmentTargets[segIdx * model.nS + j] === src) hits++;
          }
          if (!hits) continue;
          const tx = c * cellW + cellW / 2;
          const ty = gridTop + s * rowH + (rowH - 4) / 2;
          for (let p = 0; p < nP; p++) {
            const pf = Math.max(0, Math.min(1, frac * 1.2 - p * 0.15));
            const px = srcX + (tx - srcX) * pf;
            const py = topY + (ty - topY) * pf;
            ctx.globalAlpha = 0.8 * (1 - pf * 0.5);
            ctx.fillStyle = c1;
            ctx.beginPath(); ctx.arc(px, py, 2.5, 0, 2 * Math.PI); ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    // legend
    ctx.font = '11px system-ui, sans-serif';
    let lx = 4;
    const legend = [
      [c1, 'active (>= ' + actThresh.get() + ')'],
      [c3, 'matching (>= ' + learnThresh.get() + ')'],
      [muted, 'below'],
      [c1, 'dot = predictive'],
    ];
    for (const [col, label] of legend) {
      ctx.fillStyle = col;
      ctx.fillRect(lx, height - 14, 10, 10);
      ctx.fillStyle = ink;
      ctx.fillText(label, lx + 14, height - 5);
      lx += 14 + label.length * 6.5 + 16;
    }

    const nPred = predArr ? Array.from(predArr).reduce((a, b) => a + b, 0) : 0;
    hero.narration(dendriteNarration({ k: currentK, nActive: order.length, nPredictive: nPred }));
  }

  function render() {
    drawHero();
    const { nc } = model;
    const f = currentFrame;
    depolPlot.draw(
      [
        { x: Array.from({ length: nc }, (_, i) => i), y: f ? Array.from(f.depol) : new Array(nc).fill(0), label: 'depolarization', color: 'c1' },
      ],
      { xRange: [0, nc - 1], yRange: [0, 1], hlines: [{ y: depolThresh.get(), label: 'predictive threshold' }] },
    );
  }

  const rebuild = () => { model = buildModel(); precompute(); currentFrame = frames[frames.length - 1]; currentK = frames.length - 1; render(); };
  [nCells, nSegments, nSyn, actThresh, learnThresh, connPerm, seed, depolThresh, ctxSparsity].forEach((c) => c.onInput(rebuild));

  // The player time is the bit index (0..nActive); the fractional part drives
  // the arriving-bit particles. Gauges show the real value of the completed frame.
  function onFrame(t) {
    const nActive = order.length;
    const b = Math.max(0, Math.min(nActive, (t / 100) * nActive));
    currentK = b;
    currentFrame = frames[Math.floor(b)];
    render();
  }

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(render);
  precompute();
  currentFrame = frames[frames.length - 1]; // start with the full context shown
  currentK = frames.length - 1;
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
    <p>A pyramidal neuron's dendrite is split into <em>segments</em>, each carrying
    its own synapses. A segment's activation is the count of its <em>connected</em>
    synapses (permanence &ge; <code>connected_permanence</code>) whose presynaptic
    cells are active. When that count reaches <code>activation_threshold</code> the
    segment is <strong>active</strong> and depolarises the cell; at
    <code>learning_threshold</code> it is <strong>matching</strong> and eligible to
    learn.</p>
    <p>The cell's depolarization is its best segment's activation divided by the
    activation threshold, clamped to 1. A cell is <strong>predictive</strong> when
    its depolarization crosses <code>depolarization_threshold</code> &mdash; it is
    depolarised by context but not yet firing. In the Thousand Brains Theory this
    is how a neuron "expects" to fire: the matching distal segment primes it.</p>
    <p>The hero reveals the context SDR <em>bit by bit</em>. Each arriving context
    bit travels down as particles to the segments that synapse onto it; a segment's
    gauge fills as its count rises, lights up (teal) the moment it crosses
    <code>activation_threshold</code>, and the cell it belongs to becomes
    <strong>predictive</strong> (the dot above its column) once its depolarization
    crosses <code>depolarization_threshold</code>. Every count and depolarization
    shown is the real value computed at that point.</p>
  `);
  ui.equation(notes, `
    <p>activation<sub>seg</sub> = |{ synapses : connected &and; target active }|</p>
    <p>depolarization<sub>cell</sub> = min(1, max<sub>seg</sub> activation<sub>seg</sub> / activation_threshold)</p>
    <p>predictive<sub>cell</sub> = depolarization<sub>cell</sub> &ge; depolarization_threshold</p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> a dendritic segment is like a small
    conjunction detector (an AND gate over a subset of inputs) sitting in front of
    the neuron. The analogy breaks because segments learn their synapses online
    (strengthen active, weaken inactive, grow new ones) and the cell integrates the
    <em>maximum</em> segment, not a weighted sum.</p>
  `);

  ui.code(notes, () => `
import torch
from pymonntorch import Network, NeuronGroup
from conex import DendriticSegment, DendriticSegmentConfig, ContextualPrediction, TimeResolution

config = DendriticSegmentConfig(
    n_synapses_per_segment=${nSyn.get()},
    activation_threshold=${actThresh.get()},
    learning_threshold=${learnThresh.get()},
    connected_permanence=${fmt(connPerm.get())},
)
segments = DendriticSegment(
    n_segments=${nSegments.get()}, n_cells=${nCells.get()}, config=config,
)
context = ContextualPrediction(depolarization_threshold=${fmt(depolThresh.get())})

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(
    net=net, size=${nCells.get()}, behavior={100: segments, 250: context}, tag="segments",
)
net.initialize()

# wire segment 0 to a few presynaptic cells, then present a context SDR
for slot, target in enumerate([1, 2, 3]):
    neurons.segment_targets[0, slot] = target
    neurons.segment_permanences[0, slot] = 0.9
presynaptic_activity = torch.zeros(${nCells.get()}, dtype=torch.bool)
presynaptic_activity[[1, 2, 3]] = True

segments.compute_segment_activation(neurons, presynaptic_activity)
segments.compute_depolarization(neurons)
context.update_predictions(neurons, neurons.depolarization)
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

function pickDistinct(rng, n, k) {
  const pool = Array.from({ length: n }, (_, i) => i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, Math.min(k, n));
}

// At most 4 significant digits for numbers shown in text and code snippets.
function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  return Number(v.toPrecision(4)).toString();
}
