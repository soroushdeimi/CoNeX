import { DendriticSegment, ContextualPrediction } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-dendrites',
  title: 'Active dendrites',
  conex: ['DendriticSegment', 'ActiveDendriteComputation', 'ContextualPrediction'],
  source: 'conex/behaviors/neurons/numenta/active_dendrites.py',
};

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
  let lastActive = [];
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
  const rebuild = () => { model = buildModel(); render(); };
  [nCells, nSegments, nSyn, actThresh, learnThresh, connPerm, seed, depolThresh].forEach((c) => c.onInput(rebuild));
  ctxSparsity.onInput(render);

  // --- figures ---
  const segCanvas = ui.canvas(figures, { height: 260 });
  const depolPlot = ui.plot(figures, {
    height: 200, xLabel: 'cell', xUnit: 'index', yLabel: 'depolarization', yUnit: 'a.u.',
  });

  function contextSdr() {
    const rng = mulberry(seed.get() + 1000);
    const active = new Array(model.nc).fill(false);
    const nActive = Math.max(1, Math.round(model.nc * ctxSparsity.get()));
    const idx = pickDistinct(rng, model.nc, nActive);
    for (const i of idx) active[i] = true;
    return active;
  }

  function render() {
    const { seg, context, nc, ns, nS } = model;
    const active = contextSdr();
    seg.computeSegmentActivation(active);
    seg.computeDepolarization();
    context.updatePredictions(null, seg.depolarization);
    lastActive = active;

    drawSegments();
    depolPlot.draw(
      [
        { x: Array.from({ length: nc }, (_, i) => i), y: Array.from(seg.depolarization), label: 'depolarization', color: 'c1' },
      ],
      { xRange: [0, nc - 1], yRange: [0, 1], hlines: [{ y: depolThresh.get(), label: 'predictive threshold' }] },
    );
    void nS;
  }

  function drawSegments() {
    const active = lastActive;
    const { ctx, width, height } = segCanvas;
    ctx.clearRect(0, 0, width, height);
    const { seg, nc, ns } = model;
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const c5 = ui.color('c5');
    const cols = Math.min(ns, 6);
    const cellW = width / nc;
    const rowH = (height - 30) / ns;
    for (let c = 0; c < nc; c++) {
      for (let s = 0; s < ns; s++) {
        const idx = c * ns + s;
        const act = seg.segmentActivation[idx];
        const x = c * cellW + 3;
        const y = 24 + s * rowH;
        const w = cellW - 6;
        const h = rowH - 4;
        // fill by activation level
        const frac = Math.min(1, act / Math.max(1, actThresh.get()));
        ctx.globalAlpha = 0.25 + 0.75 * frac;
        ctx.fillStyle = act >= actThresh.get() ? c1 : act >= learnThresh.get() ? c3 : muted;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = muted;
        ctx.strokeRect(x, y, w, h);
        // count label
        ctx.fillStyle = ink;
        ctx.font = '11px ui-monospace, monospace';
        ctx.fillText(String(act), x + w / 2 - 4, y + h / 2 + 4);
      }
      // cell index
      ctx.fillStyle = muted;
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(String(c), c * cellW + cellW / 2 - 3, 14);
    }
    // legend
    ctx.font = '11px system-ui, sans-serif';
    let lx = 4;
    const legend = [
      [c1, 'active (>= ' + actThresh.get() + ')'],
      [c3, 'matching (>= ' + learnThresh.get() + ')'],
      [muted, 'below'],
    ];
    for (const [col, label] of legend) {
      ctx.fillStyle = col;
      ctx.fillRect(lx, height - 14, 10, 10);
      ctx.fillStyle = ink;
      ctx.fillText(label, lx + 14, height - 5);
      lx += 14 + label.length * 6.5 + 16;
    }
    void active;
    void c2; void c5;
  }

  // The segment canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  segCanvas.onResize(drawSegments);
  render();
  ui.onTheme(render);

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
    <p>Drag the context sparsity up to see more segments match; change the seed to
    draw a different context SDR and wiring.</p>
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
