// Plate: Reward-modulated STDP (SimpleRSTDP) with Dopamine and Payoff.
// Live mode (AMENDMENT 4): eligibility, dopamine and weight are stepped each
// frame. Hero (AMENDMENT 5): a synapse whose glow is the eligibility trace;
// reward events release a dopamine cloud that diffuses and turns eligibility
// into a weight change.
import * as M from '../js/models/plasticity.js';
import * as A from '../js/anim.js';

export const meta = {
  id: 'syn-rstdp',
  title: 'Reward-modulated STDP',
  conex: ['SimpleRSTDP', 'Dopamine', 'Payoff'],
  source: 'conex/behaviors/synapses/learning.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = {
    a_plus: 0.01, a_minus: 0.002, tau_c: 10, tau_s: 10, w0: 0.5,
    tau_dopamine: 10, prePeriod: 10, offset: 2, window: 5, rewardSize: 1.0,
    windowSize: 120, // ms kept on screen
  };
  const DT = 1; // ms per step

  ui.tryThis(figures, 'Try this: push post delay after pre past the reward window and watch the weight stop moving.');

  const hero = ui.hero(figures, { height: 300 });
  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'weight', yUnit: 'a.u.' });
  const dopPlot = ui.plot(figures, { height: 180, xLabel: 'time', xUnit: 'ms', yLabel: 'dopamine', yUnit: 'a.u.' });
  const eligPlot = ui.plot(figures, { height: 180, xLabel: 'time', xUnit: 'ms', yLabel: 'eligibility c', yUnit: 'a.u.' });

  let t = 0;
  let w = state.w0, c = 0, dop = 0, preT = 0, postT = 0;
  let lastPre = -1;
  const ring = []; // {t, w, c, dop, postS, inWindow, reward}
  const clouds = []; // model time of each reward (for the diffusing cloud)

  function step() {
    const preS = t % state.prePeriod === 0 ? 1 : 0;
    if (preS) lastPre = t;
    const postS = t >= state.offset && (t - state.offset) % state.prePeriod === 0 ? 1 : 0;
    const inWindow = postS && (t - lastPre) <= state.window;
    const payoff = inWindow ? state.rewardSize : 0;
    if (payoff > 0) clouds.push(t);
    dop = M.dopamineStep({ dopamine_concentration: dop, tau_dopamine: state.tau_dopamine, payoff });
    preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: DT });
    postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: DT });
    const dw = M.stdpDw({ weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus })[0];
    const s = M.rstdpStep({ c: [c], dw: [dw], tau_c: state.tau_c, dopamine_concentration: dop });
    c = s.c[0];
    w += s.dwApplied[0];
    ring.push({ t, w, c, dop, postS, inWindow, reward: payoff > 0 });
    t += DT;
    while (ring.length && ring[0].t < t - state.windowSize) ring.shift();
    while (clouds.length && clouds[0] < t - state.windowSize) clouds.shift();
  }

  function reset() {
    t = 0; w = state.w0; c = 0; dop = 0; preT = 0; postT = 0; lastPre = -1;
    ring.length = 0; clouds.length = 0;
  }

  function drawHero() {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c3 = ui.color('c3'), c4 = ui.color('c4');
    const ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const last = ring.length ? ring[ring.length - 1] : { w, c, dop, postS: 0, inWindow: false };
    let cMax = 1e-6, dopMax = 1e-6;
    for (const r of ring) { cMax = Math.max(cMax, Math.abs(r.c)); dopMax = Math.max(dopMax, r.dop); }

    const synX = W * 0.5, cy = H * 0.42;
    const preX = W * 0.18, postX = W * 0.82, r = 16;
    // axon
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(preX + r, cy); ctx.lineTo(postX - r, cy); ctx.stroke();

    // eligibility glow around the synapse (decays with tau_c)
    const elig = A.rstdpEligGlow(last.c, cMax);
    if (elig > 0.01) {
      ctx.save();
      ctx.shadowColor = c3; ctx.shadowBlur = 10 + 40 * elig;
      ctx.fillStyle = c3; ctx.globalAlpha = 0.25 + 0.5 * elig;
      ctx.beginPath(); ctx.arc(synX, cy, 14 + 22 * elig, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // dopamine cloud: diffuses outward from the synapse on each reward
    for (const ct of clouds) {
      const age = t - ct;
      const fade = A.clamp01(1 - age / (state.tau_dopamine * 2));
      if (fade <= 0) continue;
      const rad = 16 + age * 2.2;
      const g = typeof ctx.createRadialGradient === 'function' ? ctx.createRadialGradient(synX, cy, 4, synX, cy, rad) : null;
      if (g && typeof g.addColorStop === 'function') {
        g.addColorStop(0, c4); g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
      } else {
        ctx.fillStyle = c4;
      }
      ctx.save(); ctx.globalAlpha = 0.5 * fade * A.rstdpDopGlow(last.dop, dopMax);
      ctx.beginPath(); ctx.arc(synX, cy, rad, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }

    // synapse: thickness = weight
    const sw = A.stdpSynapsePx(last.w, 0, 1);
    ctx.fillStyle = c1;
    ctx.fillRect(synX - sw / 2, cy - 22, sw, 44);

    // neurons
    const neuron = (x, color, on) => {
      ctx.save(); if (on) { ctx.shadowColor = color; ctx.shadowBlur = 14; }
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, cy, r, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    };
    neuron(preX, c1, false);
    neuron(postX, c2, last.postS === 1);

    ctx.fillStyle = ink; ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('pre', preX, cy + r + 4);
    ctx.fillText('post', postX, cy + r + 4);
    ctx.textBaseline = 'bottom';
    ctx.fillText(`w = ${last.w.toFixed(3)}`, synX, cy - 30);
    ctx.fillStyle = c3; ctx.textAlign = 'left';
    ctx.fillText(`eligibility c = ${last.c.toFixed(4)}`, W * 0.05, H * 0.86);
    ctx.fillStyle = c4;
    ctx.fillText(`dopamine = ${last.dop.toFixed(4)}`, W * 0.05, H * 0.93);
  }

  function drawPlots() {
    if (!ring.length) return;
    const xs = ring.map((r) => r.t);
    let wMin = Infinity, wMax = -Infinity;
    for (const r of ring) { wMin = Math.min(wMin, r.w); wMax = Math.max(wMax, r.w); }
    weightPlot.draw([{ x: xs, y: ring.map((r) => r.w), label: 'weight', color: 'c1' }], { xRange: [t - state.windowSize, t], yRange: [wMin, wMax] });
    let dopMax = 1e-6; for (const r of ring) dopMax = Math.max(dopMax, r.dop);
    dopPlot.draw([{ x: xs, y: ring.map((r) => r.dop), label: 'dopamine', color: 'c4' }], { xRange: [t - state.windowSize, t], yRange: [0, dopMax] });
    let cMin = 0, cMax = 1e-6; for (const r of ring) { cMin = Math.min(cMin, r.c); cMax = Math.max(cMax, r.c); }
    eligPlot.draw([{ x: xs, y: ring.map((r) => r.c), label: 'eligibility c', color: 'c3' }], { xRange: [t - state.windowSize, t], yRange: [cMin, cMax] });
  }

  function onFrame() {
    drawHero();
    drawPlots();
    const last = ring.length ? ring[ring.length - 1] : { postS: 0, inWindow: false, dop: 0 };
    hero.narration(A.rstdpNarration(last));
  }

  // fill the window once so the figures are not empty before the first frame
  for (let i = 0; i < state.windowSize / DT; i++) step();
  onFrame();

  const sec = ui.section(controls, 'RSTDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; } });
  ui.slider(sec, { key: 'tau_c', label: 'tau_c (eligibility)', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_c, onInput: (v) => { state.tau_c = v; } });
  ui.slider(sec, { key: 'tau_dopamine', label: 'tau_dopamine', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_dopamine, onInput: (v) => { state.tau_dopamine = v; } });
  ui.slider(sec, { key: 'offset', label: 'post delay after pre', unit: 'ms', min: 0, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; } });
  ui.slider(sec, { key: 'window', label: 'reward window', unit: 'ms', min: 0, max: 10, step: 1, value: state.window, onInput: (v) => { state.window = v; } });
  ui.slider(sec, { key: 'rewardSize', label: 'reward size', unit: 'a.u.', min: 0, max: 3, step: 0.1, value: state.rewardSize, onInput: (v) => { state.rewardSize = v; } });
  ui.slider(sec, { key: 'w0', label: 'w0 (initial weight)', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; liveHandle && liveHandle.reset(); } });

  let liveHandle = null;
  liveHandle = ui.live(controls, {
    dt: DT, unit: 'ms', modelPerSecond: 100, windowSize: state.windowSize,
    step, reset, onFrame,
  });

  hero.onResize(onFrame);
  ui.onTheme(onFrame);

  ui.note(notes, `Reward-modulated STDP (RSTDP) splits learning into two stages. First, an <b>eligibility trace</b> <var>c</var> records recent STDP activity and decays with time constant tau_c. Second, the weight only changes when a <b>dopamine</b> signal (driven by a <b>payoff</b>/reward) is present: <var>dw</var> = <var>c</var> · dopamine. Here the reward is delivered only when the post neuron fires inside the adjustable window after the pre spike, so only causally-paired activity is reinforced. The hero shows the synapse glowing with the eligibility trace; each reward releases a dopamine cloud that diffuses outward and turns the stored eligibility into a weight change.`);
  ui.equation(notes, `<var>c</var> += −<var>c</var>/tau_c + <var>stdp_dw</var>; &nbsp; <var>w</var> += <var>c</var> · dopamine; &nbsp; dopamine += −dopamine/tau_dopamine + payoff`);
  ui.note(notes, `All quantities are dimensionless (a.u.); time is in ms. Push the post delay past the window and the reward stops being delivered, so the weight barely moves even though the same STDP activity occurred. ML analogy: reinforcement learning, where a reward (dopamine / TD error) scales a stored eligibility (policy gradient). The analogy breaks because eligibility here is a decaying trace of spike timing, not a full policy gradient.`);
  ui.code(notes, () => `from conex import SimpleRSTDP, Dopamine, Payoff, PreTrace, PostTrace
SimpleRSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c})
Dopamine(tau_dopamine=${state.tau_dopamine})
Payoff(initial_payoff=0.0)  # set network.payoff = ${state.rewardSize} when the post fires within ${state.window} ms of a pre spike`);
  return { destroy() { if (liveHandle) liveHandle.pause(); } };
}
