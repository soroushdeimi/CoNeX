// Plate: Inhibitory STDP (SimpleiSTDP) drives a real LIF postsynaptic neuron
// toward a target firing rate. Live mode (AMENDMENT 4): the LIF + iSTDP weight
// are stepped each frame. Hero (AMENDMENT 5): excitatory and inhibitory inputs
// converging on the post neuron, the inhibitory synapse thickness = weight, and
// a needle rate meter moving toward the target rho.
import * as M from '../js/models/plasticity.js';
import * as A from '../js/anim.js';
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
    f_pre: 30, f_exc: 100, I_exc: 260, I_inh: 400, tau_syn: 5, seed: 123,
    window: 500, // ms kept on screen
  };
  const DT = state.dt;

  ui.tryThis(figures, 'Try this: raise rho and watch the inhibitory weight shrink so the post rate climbs to the new target.');

  const hero = ui.hero(figures, { height: 300 });
  const ratePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'post rate', yUnit: 'Hz' });
  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'inhibitory weight', yUnit: 'a.u.' });

  const p = { ...DEFAULTS };
  let rng = M.mulberry32(state.seed);
  let v = p.v_rest;
  let w = state.w0, preT = 0, postT = 0;
  let iExc = 0, iInh = 0;
  let t = 0;
  const spikeTimes = []; // model time of each post spike
  const ring = []; // {t, rate, w}
  let lastExc = 0, lastInh = 0; // 1 if the input fired this step (for the hero flash)

  function rate() {
    const lo = t - state.window;
    let n = 0;
    for (const s of spikeTimes) if (s >= lo) n++;
    while (spikeTimes.length && spikeTimes[0] < lo) spikeTimes.shift();
    return n / (state.window / 1000);
  }

  function step() {
    const pre = rng() < (state.f_pre * DT) / 1000 ? 1 : 0;
    const exc = rng() < (state.f_exc * DT) / 1000 ? 1 : 0;
    lastInh = pre; lastExc = exc;
    const decay = Math.exp(-DT / state.tau_syn);
    iExc = iExc * decay + exc * state.I_exc;
    iInh = iInh * decay + pre * w * state.I_inh;
    const r = lifStep({ v, omega: 0 }, p, iExc - iInh, DT);
    v = r.v;
    const post = r.spike ? 1 : 0;
    if (post) spikeTimes.push(t);
    preT = M.stepTrace(preT, pre, { tau_s: state.tau_s, dt: DT });
    postT = M.stepTrace(postT, post, { tau_s: state.tau_s, dt: DT });
    const dw = M.istdpDw({
      weights: [w], pre_spike: [pre], post_spike: [post], pre_trace: [preT], post_trace: [postT],
      rho: state.rho, lr: state.lr, is_inhibitory: true, tau_s: state.tau_s,
    })[0];
    w += dw;
    ring.push({ t, rate: rate(), w });
    t += DT;
    while (ring.length && ring[0].t < t - state.window) ring.shift();
  }

  function reset() {
    rng = M.mulberry32(state.seed);
    v = p.v_rest; w = state.w0; preT = 0; postT = 0; iExc = 0; iInh = 0;
    t = 0; spikeTimes.length = 0; ring.length = 0; lastExc = 0; lastInh = 0;
  }

  function drawHero() {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c5 = ui.color('c5');
    const ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const last = ring.length ? ring[ring.length - 1] : { rate: 0, w };

    // --- converging inputs (left) ---
    const postX = W * 0.62, postY = H * 0.4, r = 20;
    const inX = W * 0.16, excY = H * 0.2, inhY = H * 0.6;
    const arrow = (x0, y0, x1, y1, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 9 * Math.cos(a - 0.4), y1 - 9 * Math.sin(a - 0.4));
      ctx.lineTo(x1 - 9 * Math.cos(a + 0.4), y1 - 9 * Math.sin(a + 0.4));
      ctx.closePath(); ctx.fill();
    };
    arrow(inX, excY, postX - r, postY - 6, c5);
    // inhibitory synapse: thickness = weight
    const sw = A.istdpSynapsePx(w, 1);
    ctx.strokeStyle = c2; ctx.lineWidth = sw;
    ctx.beginPath(); ctx.moveTo(inX, inhY); ctx.lineTo(postX - r, postY + 6); ctx.stroke();
    arrow(inX, inhY, postX - r, postY + 6, c2);
    // input nodes flash on their real spikes
    const node = (x, y, color, on) => {
      ctx.save(); if (on) { ctx.shadowColor = color; ctx.shadowBlur = 14; }
      ctx.fillStyle = on ? color : muted;
      ctx.beginPath(); ctx.arc(x, y, on ? 9 : 6, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    };
    node(inX, excY, c5, lastExc === 1);
    node(inX, inhY, c2, lastInh === 1);
    // post neuron: halo = how far above target it is firing
    const maxRate = Math.max(30, state.rho * 2);
    const over = A.clamp01(Math.max(0, last.rate - state.rho) / (maxRate - state.rho));
    ctx.save(); ctx.shadowColor = c1; ctx.shadowBlur = 6 + 24 * over;
    ctx.fillStyle = c1; ctx.beginPath(); ctx.arc(postX, postY, r, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = c5; ctx.fillText('excitatory (+)', inX, excY - 26);
    ctx.fillStyle = c2; ctx.fillText('inhibitory (−)', inX, inhY + 14);
    // weight badge on the inhibitory edge
    ctx.fillStyle = c2; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('w = ' + w.toFixed(3), (inX + postX) / 2, inhY + 10);
    ctx.fillStyle = ink; ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('post LIF', postX, postY + r + 4);

    // --- needle rate meter (bottom) ---
    const mx0 = W * 0.16, mx1 = W * 0.84, my = H * 0.85;
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(mx0, my); ctx.lineTo(mx1, my); ctx.stroke();
    // target tick at rho
    const tx = mx0 + A.istdpMeterPos(state.rho, maxRate) * (mx1 - mx0);
    ctx.strokeStyle = c5; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(tx, my - 12); ctx.lineTo(tx, my + 12); ctx.stroke();
    ctx.fillStyle = c5; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`rho = ${state.rho} Hz`, tx, my - 14);
    // needle at the current rate
    const nx = mx0 + A.istdpMeterPos(last.rate, maxRate) * (mx1 - mx0);
    ctx.strokeStyle = c1; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(nx, my); ctx.lineTo(nx, my - 22); ctx.stroke();
    ctx.fillStyle = c1; ctx.beginPath(); ctx.arc(nx, my - 22, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = ink; ctx.textBaseline = 'top';
    ctx.fillText(`rate = ${last.rate.toFixed(1)} Hz`, nx, my + 6);
  }

  function drawPlots() {
    if (!ring.length) return;
    const xs = ring.map((r) => r.t);
    let rMax = Math.max(30, state.rho * 2);
    ratePlot.draw([{ x: xs, y: ring.map((r) => r.rate), label: 'post rate', color: 'c1' }], { xRange: [t - state.window, t], yRange: [0, rMax], hlines: [{ y: state.rho, label: 'target rho' }] });
    let wMin = 0, wMax = 1;
    for (const r of ring) { wMin = Math.min(wMin, r.w); wMax = Math.max(wMax, r.w); }
    weightPlot.draw([{ x: xs, y: ring.map((r) => r.w), label: 'weight', color: 'c2' }], { xRange: [t - state.window, t], yRange: [wMin, wMax] });
  }

  function onFrame() {
    drawHero();
    drawPlots();
    const last = ring.length ? ring[ring.length - 1] : { rate: 0 };
    hero.narration(A.istdpNarration(last.rate, state.rho));
  }

  // fill the window once so the figures are not empty before the first frame
  for (let i = 0; i < Math.min(2000, Math.floor(state.window / DT)); i++) step();
  onFrame();

  const sec = ui.section(controls, 'iSTDP rule');
  ui.slider(sec, { key: 'rho', label: 'rho (target rate)', unit: 'Hz', min: 1, max: 60, step: 1, value: state.rho, onInput: (v) => { state.rho = v; } });
  ui.slider(sec, { key: 'lr', label: 'lr', unit: 'a.u.', min: 1e-5, max: 0.1, step: 1e-5, value: state.lr, onInput: (v) => { state.lr = v; } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; } });
  ui.slider(sec, { key: 'w0', label: 'w0 (initial weight)', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; liveHandle && liveHandle.reset(); } });
  ui.slider(sec, { key: 'f_pre', label: 'pre (inhibitory) rate', unit: 'Hz', min: 1, max: 100, step: 1, value: state.f_pre, onInput: (v) => { state.f_pre = v; } });
  ui.slider(sec, { key: 'f_exc', label: 'excitatory rate', unit: 'Hz', min: 10, max: 300, step: 5, value: state.f_exc, onInput: (v) => { state.f_exc = v; } });
  ui.slider(sec, { key: 'I_exc', label: 'excitatory current per spike', unit: 'pA', min: 0, max: 600, step: 10, value: state.I_exc, onInput: (v) => { state.I_exc = v; } });
  ui.slider(sec, { key: 'I_inh', label: 'inhibitory current per spike', unit: 'pA', min: 0, max: 1000, step: 10, value: state.I_inh, onInput: (v) => { state.I_inh = v; } });
  ui.slider(sec, { key: 'tau_syn', label: 'synaptic decay', unit: 'ms', min: 1, max: 20, step: 1, value: state.tau_syn, onInput: (v) => { state.tau_syn = v; } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: state.seed, onInput: (v) => { state.seed = v; liveHandle && liveHandle.reset(); } });

  let liveHandle = null;
  liveHandle = ui.live(controls, {
    dt: DT, unit: 'ms', modelPerSecond: 100, windowSize: state.window,
    step, reset, onFrame,
  });

  hero.onResize(onFrame);
  ui.onTheme(onFrame);

  ui.note(notes, `Symmetric inhibitory STDP (iSTDP) tunes the strength of inhibition so that a postsynaptic neuron settles at a target firing rate <var>rho</var>. Here the post neuron is a real CoNeX LIF driven by Poisson excitatory input and an inhibitory current through the iSTDP weight. When the post neuron fires above target the weight grows to suppress it; below target the weight shrinks. This is a homeostatic mechanism that stabilizes network activity around a set point. The hero shows the two inputs converging on the post neuron (the inhibitory synapse thickness is the weight) and a needle rate meter moving toward the target rho.`);
  ui.equation(notes, `<var>dw</var> = lr · [ <var>pre_spike</var>·(<var>alpha</var> − <var>post_trace</var>)·sign + <var>pre_trace</var>·<var>post_spike</var> ], &nbsp; <var>alpha</var> = 2·<var>rho</var>·tau_s / 1000`);
  ui.note(notes, `Input model: each excitatory spike adds a current of I_exc (pA) and each inhibitory (pre) spike adds w·I_inh (pA); both decay with time constant tau_syn (ms). The LIF is in mV/ms/pA (neurons.js, Gerstner et al. Table 6.1: R=0.5 GOhm, v_rest=−70 mV, threshold=0 mV). The values are chosen so the post rate sits in the 5-30 Hz range and the weight can move it toward rho; they are not from a specific dataset. The rate is measured over the last ${state.window} ms. ML analogy: a gain controller that adjusts a feedback weight to hold an output at a reference. The analogy breaks because the "error" is a spike trace versus a target, not a scalar loss, and the update is driven by spike timing.`);
  ui.code(notes, () => `from conex import SimpleiSTDP, PreTrace, PostTrace, LIF
SimpleiSTDP(rho=${state.rho}, lr=${state.lr}, is_inhibitory=True)
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})
LIF(R=0.5, threshold=0.0, tau=20.0, v_reset=-55.0, v_rest=-70.0)
# alpha = 2 * rho * tau_s / 1000 = ${M.istdpAlpha({ rho: state.rho, tau_s: state.tau_s }).toPrecision(4)}`);
  return { destroy() { if (liveHandle) liveHandle.pause(); } };
}
