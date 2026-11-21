// Plate: Spike-timing dependent plasticity (SimpleSTDP).
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'syn-stdp',
  title: 'Spike-timing dependent plasticity',
  conex: ['SimpleSTDP', 'PreTrace', 'PostTrace'],
  source: 'conex/behaviors/synapses/learning.py',
};

// Simulate the pre trace after a single pre spike at t=0.
function preTraceAfterSpike(tau_s, dt, steps) {
  let tr = 0;
  const out = [0];
  for (let t = 0; t < steps; t++) {
    const spike = t === 0 ? 1 : 0;
    tr = M.stepTrace(tr, spike, { tau_s, dt });
    out.push(tr);
  }
  return out; // out[t] = trace after step t
}

// STDP window: dw for a single pre spike and a single post spike separated by delta_t.
function stdpWindow({ a_plus, a_minus, tau_s, dt, bound, w_min, w_max, w }) {
  const maxT = 60;
  const pre = preTraceAfterSpike(tau_s, dt, maxT);
  const post = preTraceAfterSpike(tau_s, dt, maxT);
  const b = M.BOUNDS[bound] || M.noBound;
  const pts = [];
  for (let d = -maxT; d <= maxT; d++) {
    let dw = 0;
    if (d >= 0) dw = pre[d] * 1 * a_plus * b(w, w_min, w_max); // pre at 0, post at d
    else dw = -1 * post[-d] * a_minus * b(w, w_min, w_max); // post at 0, pre at -d
    pts.push([d, dw]);
  }
  return pts;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = {
    a_plus: 0.01, a_minus: 0.002, tau_s: 10, dt: 1,
    w_min: 0, w_max: 1, w0: 0.5, bound: 'no_bound', offset: 5, steps: 40,
  };

  const windowPlot = ui.plot(figures, { height: 200, xLabel: 'delta t', xUnit: 'ms', yLabel: 'dw', yUnit: 'a.u.' });
  const tracePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'trace / weight', yUnit: 'a.u.' });
  const diag = ui.canvas(figures, { height: 200 });

  let run = null;   // { preS, postS, preT, postT, w, steps }
  let win = null;   // [[deltaT, dw], ...]
  let curT = 0;
  let player = null;

  // Run the pre/post trains once; store traces and weight per step.
  function computeRun() {
    const T = state.steps;
    const preS = [], postS = [];
    for (let t = 0; t < T; t++) { preS.push(t % 10 === 0 ? 1 : 0); postS.push((t - state.offset) % 10 === 0 && t >= state.offset ? 1 : 0); }
    let preT = 0, postT = 0, w = state.w0;
    const preTs = [], postTs = [], ws = [];
    for (let t = 0; t < T; t++) {
      preT = M.stepTrace(preT, preS[t], { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS[t], { tau_s: state.tau_s, dt: state.dt });
      const dw = M.stdpDw({ weights: [w], pre_spike: [preS[t]], post_spike: [postS[t]], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus, w_min: state.w_min, w_max: state.w_max, positive_bound: state.bound, negative_bound: state.bound })[0];
      w += dw;
      preTs.push(preT); postTs.push(postT); ws.push(w);
    }
    // yRange from the whole run's data (traces can exceed 1, weight can go < 0) with 10% headroom
    const dataMax = Math.max(state.w_max, ...preTs, ...postTs, ...ws);
    const dataMin = Math.min(state.w_min, ...preTs, ...postTs, ...ws);
    run = { preS, postS, preT: preTs, postT: postTs, w: ws, steps: T, yMin: dataMin, yMax: dataMax * 1.1 };
  }

  function computeWindow() {
    win = stdpWindow({ ...state, w: state.w0 });
  }

  // The pre -> synapse -> post diagram at time t: traces as glowing bars,
  // weight as the synapse thickness, a spike dot travelling along the axon.
  function drawDiagram(t) {
    const ctx = diag.ctx, W = diag.width, H = diag.height;
    const pal = { c1: ui.color('c1'), c2: ui.color('c2'), c4: ui.color('c4'), ink: ui.color('ink'), muted: ui.color('muted') };
    ctx.clearRect(0, 0, W, H);
    const idx = Math.max(0, Math.min(Math.floor(t), run.steps - 1));
    const frac = t - Math.floor(t);
    const preTrace = run.preT[idx], postTrace = run.postT[idx], w = run.w[idx];
    const preX = W * 0.12, postX = W * 0.88, synX = W * 0.5, cy = H * 0.5, r = 16;
    const baseY = H * 0.82;

    // axon
    ctx.strokeStyle = pal.muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(preX + r, cy); ctx.lineTo(postX - r, cy); ctx.stroke();

    // synapse: thickness = weight
    const sw = 3 + w * 14;
    ctx.fillStyle = pal.c4;
    ctx.fillRect(synX - sw / 2, cy - 20, sw, 40);

    // trace bars (glowing)
    const bar = (x, val, color) => {
      const h = Math.max(0, val) * (H * 0.35);
      ctx.save(); ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fillStyle = color;
      ctx.fillRect(x - 4, baseY - h, 8, h); ctx.restore();
    };
    bar(W * 0.28, preTrace, pal.c1);
    bar(W * 0.72, postTrace, pal.c2);

    // spike dots travelling along the axon (schematic timing)
    const dot = (x, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, cy, 4, 0, Math.PI * 2); ctx.fill(); };
    if (run.preS[idx]) dot(preX + r + frac * (synX - (preX + r)), pal.c1);
    if (run.postS[idx]) dot(synX + frac * ((postX - r) - synX), pal.c2);

    // neurons
    const neuron = (x, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, cy, r, 0, Math.PI * 2); ctx.fill(); };
    neuron(preX, pal.c1); neuron(postX, pal.c2);

    // labels
    ctx.fillStyle = pal.ink;
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('pre', preX, cy + r + 4);
    ctx.fillText('post', postX, cy + r + 4);
    ctx.textBaseline = 'bottom';
    ctx.fillText(`w = ${w.toFixed(3)}`, synX, cy - 24);
    ctx.textBaseline = 'top';
    ctx.fillText('pre trace', W * 0.28, baseY + 4);
    ctx.fillText('post trace', W * 0.72, baseY + 4);
  }

  function renderAt(t) {
    if (!run) return;
    curT = t;
    const T = run.steps;
    tracePlot.draw([
      { x: run.w.map((_, i) => i), y: run.preT, label: 'pre_trace', color: 'c1' },
      { x: run.w.map((_, i) => i), y: run.postT, label: 'post_trace', color: 'c2' },
      { x: run.w.map((_, i) => i), y: run.w, label: 'weight', color: 'c4' },
    ], { xRange: [0, T], yRange: [run.yMin, run.yMax], hlines: [{ y: state.w_min, label: 'w_min' }, { y: state.w_max, label: 'w_max' }], revealX: t });
    // window sweeps delta t from -60 to +60 as the run plays
    const maxT = 60;
    const revealWin = -maxT + (t / T) * (2 * maxT);
    windowPlot.draw([{ x: win.map((p) => p[0]), y: win.map((p) => p[1]), label: 'STDP window', color: 'c1' }], { xRange: [-maxT, maxT], hlines: [{ y: 0, label: '0' }], revealX: revealWin });
    drawDiagram(t);
  }

  function recompute() {
    computeRun();
    computeWindow();
    if (player) { player.setDuration(run.steps); renderAt(Math.min(curT, run.steps)); }
    else renderAt(run.steps); // no player (checker): show the full run
  }

  const sec = ui.section(controls, 'STDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; recompute(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; recompute(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; recompute(); } });
  ui.slider(sec, { key: 'w0', label: 'w0', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; recompute(); } });
  ui.slider(sec, { key: 'offset', label: 'post offset', unit: 'steps', min: -10, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; recompute(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'steps', min: 20, max: 100, step: 10, value: state.steps, onInput: (v) => { state.steps = v; recompute(); } });
  ui.select(sec, { key: 'bound', label: 'bound', value: state.bound, options: [{ value: 'no_bound', label: 'no_bound' }, { value: 'hard_bound', label: 'hard_bound' }, { value: 'soft_bound', label: 'soft_bound' }], onInput: (v) => { state.bound = v; recompute(); } });

  recompute();
  // the cross-area checker's mock ui has no player; fall back to the full view
  player = (typeof ui.player === 'function') ? ui.player(controls, {
    duration: run.steps, unit: 'ms', playSeconds: 5, loop: true,
    onFrame: (t) => renderAt(t),
  }) : null;
  if (player) player.seek(0); // start empty; the player then reveals the run

  if (typeof diag.onResize === 'function') diag.onResize(() => drawDiagram(curT));
  ui.onTheme(() => renderAt(curT));

  ui.note(notes, `STDP changes a synapse weight depending on the timing of pre- and post-synaptic spikes. If the pre neuron fires just before the post neuron (causal pairing) the weight is potentiated; if the post fires first (anti-causal) it is depressed. This is the canonical associative learning rule for spiking networks and is thought to underlie Hebbian learning in cortex. The diagram shows the pre neuron, the synapse (its thickness is the weight) and the post neuron; the glowing bars are the pre/post traces and the dots are spikes travelling along the axon (schematic timing).`);
  ui.equation(notes, `<var>dw</var> = <var>a</var><sub>+</sub>·<var>pre_trace</var>·<var>post_spike</var> − <var>a</var><sub>−</sub>·<var>pre_spike</var>·<var>post_trace</var>`);
  ui.note(notes, `The traces decay with time constant tau_s (ms); the weight is dimensionless (a.u.). The window plot simulates single spike pairs to show dw versus the timing offset. ML analogy: this is like a Hebbian "fire together, wire together" update, but gated by precise timing. The analogy breaks because classical Hebbian learning uses rates, not spike timing, and has no depression term.`);
  ui.code(notes, () => `from conex import SimpleSTDP, PreTrace, PostTrace, PreSpikeCatcher, PostSpikeCatcher
SimpleSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, w_min=${state.w_min}, w_max=${state.w_max},
           positive_bound="${state.bound}", negative_bound="${state.bound}")
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})`);
  return { destroy() { if (player) player.pause(); } };
}
