// Plate: Spike-timing dependent plasticity (SimpleSTDP).
// Live mode (AMENDMENT 4): the model is stepped each frame like an oscilloscope.
// Hero (AMENDMENT 5): pre neuron -> axon -> synapse -> post neuron; halo = trace,
// synapse thickness = weight, a spike dot on each firing, a +dw/-dw burst on each
// pairing, and the matching point of the STDP window curve lit up.
import * as M from '../js/models/plasticity.js';
import * as A from '../js/anim.js';

export const meta = {
  id: 'syn-stdp',
  title: 'Spike-timing dependent plasticity',
  conex: ['SimpleSTDP', 'PreTrace', 'PostTrace'],
  source: 'conex/behaviors/synapses/learning.py',
};

// Simulate the pre trace after a single pre spike at t=0 (for the window curve).
function preTraceAfterSpike(tau_s, dt, steps) {
  let tr = 0;
  const out = [0];
  for (let t = 0; t < steps; t++) {
    tr = M.stepTrace(tr, t === 0 ? 1 : 0, { tau_s, dt });
    out.push(tr);
  }
  return out;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = {
    a_plus: 0.01, a_minus: 0.002, tau_s: 10,
    w_min: 0, w_max: 1, w0: 0.5, bound: 'no_bound', offset: 5, window: 60,
  };
  const DT = 1; // ms of model time per step

  const hero = ui.hero(figures, { height: 300 });
  const tracePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'trace / weight', yUnit: 'a.u.' });
  const windowPlot = ui.plot(figures, { height: 200, xLabel: 'delta t', xUnit: 'ms', yLabel: 'dw', yUnit: 'a.u.' });

  // Live model state (one dt per step).
  let t = 0;
  let preT = 0, postT = 0, w = state.w0;
  let lastPre = -1, lastPost = -1;
  const ring = []; // {t, preT, postT, w, preS, postS, dw}
  let windowBase = null; // [[deltaT, unbounded dw], ...]

  function computeWindowBase() {
    const maxT = 60;
    const pre = preTraceAfterSpike(state.tau_s, DT, maxT);
    const post = preTraceAfterSpike(state.tau_s, DT, maxT);
    windowBase = [];
    for (let d = -maxT; d <= maxT; d++) {
      const dw = d >= 0 ? pre[d] * state.a_plus : -post[-d] * state.a_minus;
      windowBase.push([d, dw]);
    }
  }

  function step() {
    const preS = t % 10 === 0 ? 1 : 0;
    const postS = (t - state.offset) % 10 === 0 && t >= state.offset ? 1 : 0;
    preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: DT });
    postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: DT });
    const dw = M.stdpDw({
      weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT],
      a_plus: state.a_plus, a_minus: state.a_minus, w_min: state.w_min, w_max: state.w_max,
      positive_bound: state.bound, negative_bound: state.bound,
    })[0];
    w += dw;
    if (preS) lastPre = t;
    if (postS) lastPost = t;
    ring.push({ t, preT, postT, w, preS, postS, dw });
    t += DT;
    while (ring.length && ring[0].t < t - state.window) ring.shift();
  }

  function reset() {
    t = 0; preT = 0; postT = 0; w = state.w0; lastPre = -1; lastPost = -1; ring.length = 0;
  }

  // The neuron diagram + a compact STDP window curve with the lit point.
  function drawHero() {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c4 = ui.color('c4');
    const ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const last = ring.length ? ring[ring.length - 1] : { preT: 0, postT: 0, w, preS: 0, postS: 0, dw: 0 };

    // --- neuron diagram (top) ---
    const cy = H * 0.32, r = 18;
    const preX = W * 0.15, postX = W * 0.85, synX = W * 0.5;
    // axon
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(preX + r, cy); ctx.lineTo(postX - r, cy); ctx.stroke();
    // synapse: thickness = weight
    const sw = A.stdpSynapsePx(last.w, state.w_min, state.w_max);
    ctx.fillStyle = c4;
    ctx.fillRect(synX - sw / 2, cy - 22, sw, 44);
    // neurons with a halo whose brightness is the trace
    const neuron = (x, color, halo) => {
      ctx.save();
      ctx.shadowColor = color; ctx.shadowBlur = 6 + 26 * halo;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, cy, r, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    };
    neuron(preX, c1, A.stdpHalo(last.preT));
    neuron(postX, c2, A.stdpHalo(last.postT));
    // a spike dot on the neuron that just fired
    const dot = (x, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, cy, 5, 0, Math.PI * 2); ctx.fill(); };
    if (last.preS) dot(preX, c1);
    if (last.postS) dot(postX, c2);
    // +dw / -dw burst near the synapse
    if (Math.abs(last.dw) > 1e-9) {
      ctx.fillStyle = last.dw > 0 ? c1 : c2;
      ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(`${last.dw > 0 ? '+' : ''}${last.dw.toFixed(4)}`, synX, cy - 26);
    }
    // labels
    ctx.fillStyle = ink; ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('pre', preX, cy + r + 4);
    ctx.fillText('post', postX, cy + r + 4);
    ctx.textBaseline = 'bottom';
    ctx.fillText(`w = ${last.w.toFixed(3)}`, synX, cy - 40);

    // --- STDP window curve (bottom) with the lit point ---
    const maxT = 60;
    const wx0 = W * 0.2, wx1 = W * 0.8, wy0 = H * 0.62, wy1 = H * 0.9;
    const b = (M.BOUNDS[state.bound] || M.noBound)(last.w, state.w_min, state.w_max);
    let dwMax = 1e-6;
    for (const p of windowBase) dwMax = Math.max(dwMax, Math.abs(p[1]) * b);
    const sx = (d) => wx0 + ((d + maxT) / (2 * maxT)) * (wx1 - wx0);
    const sy = (dw) => (wy0 + wy1) / 2 - (dw / dwMax) * ((wy1 - wy0) / 2);
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx(0), wy0); ctx.lineTo(sx(0), wy1); ctx.stroke(); // delta t = 0
    ctx.strokeStyle = c1; ctx.lineWidth = 1.5;
    ctx.beginPath();
    windowBase.forEach((p, i) => { const x = sx(p[0]), y = sy(p[1] * b); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    // the lit point for the current pairing
    const d = A.stdpHighlightDelta(t - 1, lastPre, lastPost, last.preS, last.postS);
    if (d !== null) {
      const dw = (d >= 0 ? windowBase[d + maxT][1] * state.a_plus : windowBase[d + maxT][1] * state.a_minus) * b;
      ctx.save(); ctx.shadowColor = c2; ctx.shadowBlur = 12; ctx.fillStyle = c2;
      ctx.beginPath(); ctx.arc(sx(d), sy(dw), 5, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
    ctx.fillStyle = muted; ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('STDP window (dw vs delta t)', (wx0 + wx1) / 2, wy1 + 2);
  }

  function drawPlots() {
    if (!ring.length) return;
    const xs = ring.map((r) => r.t);
    let yMin = Infinity, yMax = -Infinity;
    for (const r of ring) {
      yMin = Math.min(yMin, r.preT, r.postT, r.w);
      yMax = Math.max(yMax, r.preT, r.postT, r.w);
    }
    yMin = Math.min(yMin, state.w_min); yMax = Math.max(yMax, state.w_max);
    tracePlot.draw([
      { x: xs, y: ring.map((r) => r.preT), label: 'pre_trace', color: 'c1' },
      { x: xs, y: ring.map((r) => r.postT), label: 'post_trace', color: 'c2' },
      { x: xs, y: ring.map((r) => r.w), label: 'weight', color: 'c4' },
    ], { xRange: [t - state.window, t], yRange: [yMin, yMax * 1.05], hlines: [{ y: state.w_min, label: 'w_min' }, { y: state.w_max, label: 'w_max' }] });
    const b = (M.BOUNDS[state.bound] || M.noBound)(w, state.w_min, state.w_max);
    windowPlot.draw([{ x: windowBase.map((p) => p[0]), y: windowBase.map((p) => p[1] * b), label: 'STDP window', color: 'c1' }], { xRange: [-60, 60], hlines: [{ y: 0, label: '0' }] });
  }

  function onFrame() {
    drawHero();
    drawPlots();
    const last = ring.length ? ring[ring.length - 1] : { preS: 0, postS: 0, dw: 0 };
    hero.narration(A.stdpNarration(last));
  }

  computeWindowBase();
  // fill the window once so the figures are not empty before the first frame
  for (let i = 0; i < state.window / DT; i++) step();
  onFrame();

  const sec = ui.section(controls, 'STDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; computeWindowBase(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; computeWindowBase(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; computeWindowBase(); } });
  ui.slider(sec, { key: 'w0', label: 'w0 (initial weight)', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; liveHandle && liveHandle.reset(); } });
  ui.slider(sec, { key: 'offset', label: 'post offset', unit: 'steps', min: -10, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; } });
  ui.slider(sec, { key: 'window', label: 'rolling window', unit: 'ms', min: 20, max: 120, step: 10, value: state.window, onInput: (v) => { state.window = v; } });
  ui.select(sec, { key: 'bound', label: 'bound', value: state.bound, options: [{ value: 'no_bound', label: 'no_bound' }, { value: 'hard_bound', label: 'hard_bound' }, { value: 'soft_bound', label: 'soft_bound' }], onInput: (v) => { state.bound = v; } });

  let liveHandle = null;
  liveHandle = ui.live(controls, {
    dt: DT, unit: 'ms', modelPerSecond: 100, windowSize: state.window,
    step, reset, onFrame,
  });

  hero.onResize(onFrame);
  ui.onTheme(onFrame);

  ui.note(notes, `STDP changes a synapse weight depending on the timing of pre- and post-synaptic spikes. If the pre neuron fires just before the post neuron (causal pairing) the weight is potentiated; if the post fires first (anti-causal) it is depressed. This is the canonical associative learning rule for spiking networks and is thought to underlie Hebbian learning in cortex. The hero shows the pre neuron, the synapse (its thickness is the weight) and the post neuron; each neuron's halo is its spike trace, the dots are spikes, and the lit point on the window curve is the current pairing.`);
  ui.equation(notes, `<var>dw</var> = <var>a</var><sub>+</sub>·<var>pre_trace</var>·<var>post_spike</var> − <var>a</var><sub>−</sub>·<var>pre_spike</var>·<var>post_trace</var>`);
  ui.note(notes, `The traces decay with time constant tau_s (ms); the weight is dimensionless (a.u.). The window curve simulates single spike pairs to show dw versus the timing offset. ML analogy: this is like a Hebbian "fire together, wire together" update, but gated by precise timing. The analogy breaks because classical Hebbian learning uses rates, not spike timing, and has no depression term.`);
  ui.code(notes, () => `from conex import SimpleSTDP, PreTrace, PostTrace, PreSpikeCatcher, PostSpikeCatcher
SimpleSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, w_min=${state.w_min}, w_max=${state.w_max},
           positive_bound="${state.bound}", negative_bound="${state.bound}")
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})`);
  return { destroy() { if (liveHandle) liveHandle.pause(); } };
}
