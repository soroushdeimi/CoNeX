// Plate: Inhibitory STDP (SimpleiSTDP) drives a real LIF postsynaptic neuron
// toward a target firing rate. The post neuron is a CoNeX LIF (neurons.js) driven
// by Poisson excitatory input and an inhibitory current through the iSTDP weight.
import * as M from '../js/models/plasticity.js';
import { lifStep, DEFAULTS } from '../js/models/neurons.js';

export const meta = {
  id: 'syn-istdp',
  title: 'Inhibitory STDP',
  conex: ['SimpleiSTDP', 'LIF'],
  source: 'conex/behaviors/synapses/learning.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  // Physiological units (neurons.js): v in mV, t in ms, I in pA, R in GOhm.
  const state = {
    rho: 20, lr: 0.01, tau_s: 10, dt: 0.1, w0: 0.3,
    f_pre: 30, f_exc: 100, I_exc: 260, I_inh: 400, tau_syn: 5, steps: 30000, seed: 123,
  };

  const ratePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'post rate', yUnit: 'Hz' });
  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'inhibitory weight', yUnit: 'a.u.' });
  const diag = ui.canvas(figures, { height: 160 });

  let run = null;   // { xs, rates, ws, preS, excS, postS, tMax }
  let curT = 0;
  let player = null;

  function computeRun() {
    const p = { ...DEFAULTS };
    const rng = M.mulberry32(state.seed);
    let v = p.v_rest;
    let w = state.w0, preT = 0, postT = 0, spikes = 0;
    let iExc = 0, iInh = 0;
    const decay = Math.exp(-state.dt / state.tau_syn);
    const xs = [], rates = [], ws = [], preS = [], excS = [], postS = [];
    const T = state.steps;
    for (let t = 0; t < T; t++) {
      const pre = rng() < (state.f_pre * state.dt) / 1000 ? 1 : 0;
      const exc = rng() < (state.f_exc * state.dt) / 1000 ? 1 : 0;
      // each spike adds a current that decays with tau_syn (a.u. of pA)
      iExc = iExc * decay + exc * state.I_exc;
      iInh = iInh * decay + pre * w * state.I_inh;
      const I = iExc - iInh; // pA
      const r = lifStep({ v, omega: 0 }, p, I, state.dt);
      v = r.v;
      const post = r.spike ? 1 : 0;
      spikes += post;
      preT = M.stepTrace(preT, pre, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, post, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.istdpDw({
        weights: [w], pre_spike: [pre], post_spike: [post], pre_trace: [preT], post_trace: [postT],
        rho: state.rho, lr: state.lr, is_inhibitory: true, tau_s: state.tau_s,
      })[0];
      w += dw;
      preS.push(pre); excS.push(exc); postS.push(post);
      if (t > 0 && t % 100 === 0) {
        xs.push(t * state.dt);
        rates.push(spikes / ((t * state.dt) / 1000));
        ws.push(w);
      }
    }
    const alpha = M.istdpAlpha({ rho: state.rho, tau_s: state.tau_s });
    run = { xs, rates, ws, preS, excS, postS, alpha, tMax: T * state.dt };
    state._alpha = alpha;
    state._finalRate = spikes / ((T * state.dt) / 1000);
  }

  // Small diagram: excitatory (c5) and inhibitory (c2) inputs flash on their
  // real spikes, feeding the post neuron (c1).
  function drawDiagram(t) {
    const ctx = diag.ctx, W = diag.width, H = diag.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c5 = ui.color('c5'), ink = ui.color('ink');
    ctx.clearRect(0, 0, W, H);
    const idx = Math.max(0, Math.min(Math.floor(t / state.dt), run.preS.length - 1));
    const postX = W * 0.7, postY = H * 0.5, r = 14;
    const excY = H * 0.22, inhY = H * 0.78, inX = W * 0.15;
    ctx.strokeStyle = ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(inX, excY); ctx.lineTo(postX - r, postY - 4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(inX, inhY); ctx.lineTo(postX - r, postY + 4); ctx.stroke();
    const node = (x, y, color, on) => {
      ctx.save(); if (on) { ctx.shadowColor = color; ctx.shadowBlur = 12; }
      ctx.fillStyle = on ? color : ink; ctx.beginPath(); ctx.arc(x, y, on ? 8 : 5, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    };
    node(inX, excY, c5, run.excS[idx] === 1);
    node(inX, inhY, c2, run.preS[idx] === 1);
    node(postX, postY, c1, run.postS[idx] === 1);
  }

  function renderAt(t) {
    if (!run) return;
    curT = t;
    ratePlot.draw([{ x: run.xs, y: run.rates, label: 'post rate', color: 'c1' }], { xRange: [0, run.tMax], hlines: [{ y: state.rho, label: 'target rho' }], revealX: t });
    weightPlot.draw([{ x: run.xs, y: run.ws, label: 'weight', color: 'c2' }], { xRange: [0, run.tMax], hlines: [{ y: run.alpha, label: 'alpha' }], revealX: t });
    drawDiagram(t);
  }

  function recompute() {
    computeRun();
    if (player) { player.setDuration(run.tMax); renderAt(Math.min(curT, run.tMax)); }
    else renderAt(run.tMax); // no player (checker): show the full run
  }

  const sec = ui.section(controls, 'iSTDP rule');
  ui.slider(sec, { key: 'rho', label: 'rho (target rate)', unit: 'Hz', min: 1, max: 60, step: 1, value: state.rho, onInput: (v) => { state.rho = v; recompute(); } });
  ui.slider(sec, { key: 'lr', label: 'lr', unit: 'a.u.', min: 1e-5, max: 0.1, step: 1e-5, value: state.lr, onInput: (v) => { state.lr = v; recompute(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; recompute(); } });
  ui.slider(sec, { key: 'w0', label: 'w0', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; recompute(); } });
  ui.slider(sec, { key: 'f_pre', label: 'pre (inhibitory) rate', unit: 'Hz', min: 1, max: 100, step: 1, value: state.f_pre, onInput: (v) => { state.f_pre = v; recompute(); } });
  ui.slider(sec, { key: 'f_exc', label: 'excitatory rate', unit: 'Hz', min: 10, max: 300, step: 5, value: state.f_exc, onInput: (v) => { state.f_exc = v; recompute(); } });
  ui.slider(sec, { key: 'I_exc', label: 'excitatory current per spike', unit: 'pA', min: 0, max: 600, step: 10, value: state.I_exc, onInput: (v) => { state.I_exc = v; recompute(); } });
  ui.slider(sec, { key: 'I_inh', label: 'inhibitory current per spike', unit: 'pA', min: 0, max: 1000, step: 10, value: state.I_inh, onInput: (v) => { state.I_inh = v; recompute(); } });
  ui.slider(sec, { key: 'tau_syn', label: 'synaptic decay', unit: 'ms', min: 1, max: 20, step: 1, value: state.tau_syn, onInput: (v) => { state.tau_syn = v; recompute(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'steps', min: 5000, max: 100000, step: 5000, value: state.steps, onInput: (v) => { state.steps = v; recompute(); } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: state.seed, onInput: (v) => { state.seed = v; recompute(); } });

  recompute();
  // the cross-area checker's mock ui has no player; fall back to the full view
  player = (typeof ui.player === 'function') ? ui.player(controls, {
    duration: run.tMax, unit: 'ms', playSeconds: 5, loop: true,
    onFrame: (t) => renderAt(t),
  }) : null;
  if (player) player.seek(0); // start empty; the player then reveals the run
  if (typeof diag.onResize === 'function') diag.onResize(() => drawDiagram(curT));
  ui.onTheme(() => renderAt(curT));

  ui.note(notes, `Symmetric inhibitory STDP (iSTDP) tunes the strength of inhibition so that a postsynaptic neuron settles at a target firing rate <var>rho</var>. Here the post neuron is a real CoNeX LIF driven by Poisson excitatory input and an inhibitory current through the iSTDP weight. When the post neuron fires above target the weight grows to suppress it; below target the weight shrinks. This is a homeostatic mechanism that stabilizes network activity around a set point.`);
  ui.equation(notes, `<var>dw</var> = lr · [ <var>pre_spike</var>·(<var>alpha</var> − <var>post_trace</var>)·sign + <var>pre_trace</var>·<var>post_spike</var> ], &nbsp; <var>alpha</var> = 2·<var>rho</var>·tau_s / 1000`);
  ui.note(notes, `Input model: each excitatory spike adds a current of I_exc (pA) and each inhibitory (pre) spike adds w·I_inh (pA); both decay with time constant tau_syn (ms). The LIF is in mV/ms/pA (neurons.js, Gerstner et al. Table 6.1: R=0.5 GOhm, v_rest=−70 mV, threshold=0 mV). The values are chosen so the post rate sits in the 5-30 Hz range and the weight can move it toward rho; they are not from a specific dataset. The <var>steps</var> control is a step count, not a duration: at dt = 0.1 ms the default 30000 steps is 3000 ms (3 s). The top plot shows the post rate approaching the target rho; the bottom plot shows the weight trace. The diagram flashes the excitatory (green) and inhibitory (coral) inputs on their real spikes feeding the post neuron (teal).`);
  ui.note(notes, `ML analogy: a gain controller that adjusts a feedback weight to hold an output at a reference. The analogy breaks because the "error" is a spike trace versus a target, not a scalar loss, and the update is driven by spike timing.`);
  ui.code(notes, () => `from conex import SimpleiSTDP, PreTrace, PostTrace, LIF
SimpleiSTDP(rho=${state.rho}, lr=${state.lr}, is_inhibitory=True)
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})
LIF(R=0.5, threshold=0.0, tau=20.0, v_reset=-55.0, v_rest=-70.0)
# alpha = 2 * rho * tau_s / 1000 = ${state._alpha.toPrecision(4)}`);
  return { destroy() { if (player) player.pause(); } };
}
