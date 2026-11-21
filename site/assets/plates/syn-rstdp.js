// Plate: Reward-modulated STDP (SimpleRSTDP) with Dopamine and Payoff.
// Reward is delivered only when the post neuron fires inside an adjustable
// window after the pre spike.
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'syn-rstdp',
  title: 'Reward-modulated STDP',
  conex: ['SimpleRSTDP', 'Dopamine', 'Payoff'],
  source: 'conex/behaviors/synapses/learning.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = {
    a_plus: 0.01, a_minus: 0.002, tau_c: 10, tau_s: 10, dt: 1, w0: 0.5,
    tau_dopamine: 10, steps: 60, prePeriod: 10, offset: 2, window: 5, rewardSize: 1.0,
  };

  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'weight', yUnit: 'a.u.' });
  const dopPlot = ui.plot(figures, { height: 180, xLabel: 'time', xUnit: 'ms', yLabel: 'dopamine', yUnit: 'a.u.' });
  const eligPlot = ui.plot(figures, { height: 180, xLabel: 'time', xUnit: 'ms', yLabel: 'eligibility c', yUnit: 'a.u.' });
  const timeline = ui.canvas(figures, { height: 60 });

  let run = null;   // { xs, ws, cs, dops, noW, rewards, T }
  let curT = 0;
  let player = null;

  function simulate(rewardOn) {
    const T = state.steps;
    let w = state.w0, c = [0], dop = 0, preT = 0, postT = 0;
    let lastPre = -1e9;
    const xs = [], ws = [], cs = [], dops = [], rewards = [];
    for (let t = 0; t < T; t++) {
      const preS = t % state.prePeriod === 0 ? 1 : 0;
      if (preS) lastPre = t;
      const postS = t >= state.offset && (t - state.offset) % state.prePeriod === 0 ? 1 : 0;
      // reward only when the post fires inside `window` steps after a pre spike
      const payoff = rewardOn && postS && (t - lastPre) <= state.window ? state.rewardSize : 0;
      if (payoff > 0) rewards.push(t);
      dop = M.dopamineStep({ dopamine_concentration: dop, tau_dopamine: state.tau_dopamine, payoff });
      preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.stdpDw({ weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus });
      const step = M.rstdpStep({ c, dw, tau_c: state.tau_c, dopamine_concentration: dop });
      c = step.c;
      w += step.dwApplied[0];
      xs.push(t); ws.push(w); cs.push(c[0]); dops.push(dop);
    }
    return { xs, ws, cs, dops, rewards, T };
  }

  // Reward events flash on a timeline as the playhead passes them.
  function drawTimeline(t) {
    const ctx = timeline.ctx, W = timeline.width, H = timeline.height;
    const ink = ui.color('ink'), c3 = ui.color('c3'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const x0 = 40, x1 = W - 12, y = H * 0.5;
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    const sx = (x) => x0 + (x / run.T) * (x1 - x0);
    for (const rt of run.rewards) {
      const on = rt <= t;
      ctx.save(); if (on) { ctx.shadowColor = c3; ctx.shadowBlur = 10; }
      ctx.fillStyle = on ? c3 : muted;
      ctx.beginPath(); ctx.arc(sx(rt), y, on ? 6 : 3, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
    // playhead
    ctx.strokeStyle = ink; ctx.beginPath(); ctx.moveTo(sx(t), 6); ctx.lineTo(sx(t), H - 6); ctx.stroke();
  }

  function renderAt(t) {
    if (!run) return;
    curT = t;
    const T = run.T;
    weightPlot.draw([
      { x: run.xs, y: run.ws, label: 'weight (reward in window)', color: 'c1' },
      { x: run.xs, y: run.noW, label: 'weight (no reward)', color: 'c2' },
    ], { xRange: [0, T], revealX: t });
    dopPlot.draw([{ x: run.xs, y: run.dops, label: 'dopamine', color: 'c4' }], { xRange: [0, T], yRange: [0, Math.max(0.1, ...run.dops)], revealX: t });
    // eligibility can go negative (the a_minus term), so span both signs
    const cMin = Math.min(0, ...run.cs), cMax = Math.max(0.01, ...run.cs);
    eligPlot.draw([{ x: run.xs, y: run.cs, label: 'eligibility c', color: 'c3' }], { xRange: [0, T], yRange: [cMin, cMax], revealX: t });
    drawTimeline(t);
  }

  function recompute() {
    const withR = simulate(true);
    const noR = simulate(false);
    run = { ...withR, noW: noR.ws };
    if (player) { player.setDuration(run.T); renderAt(Math.min(curT, run.T)); }
    else renderAt(run.T); // no player (checker): show the full run
  }

  const sec = ui.section(controls, 'RSTDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; recompute(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; recompute(); } });
  ui.slider(sec, { key: 'tau_c', label: 'tau_c (eligibility)', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_c, onInput: (v) => { state.tau_c = v; recompute(); } });
  ui.slider(sec, { key: 'tau_dopamine', label: 'tau_dopamine', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_dopamine, onInput: (v) => { state.tau_dopamine = v; recompute(); } });
  ui.slider(sec, { key: 'offset', label: 'post delay after pre', unit: 'ms', min: 0, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; recompute(); } });
  ui.slider(sec, { key: 'window', label: 'reward window', unit: 'ms', min: 0, max: 10, step: 1, value: state.window, onInput: (v) => { state.window = v; recompute(); } });
  ui.slider(sec, { key: 'rewardSize', label: 'reward size', unit: 'a.u.', min: 0, max: 3, step: 0.1, value: state.rewardSize, onInput: (v) => { state.rewardSize = v; recompute(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'ms', min: 20, max: 200, step: 10, value: state.steps, onInput: (v) => { state.steps = v; recompute(); } });

  recompute();
  // the cross-area checker's mock ui has no player; fall back to the full view
  player = (typeof ui.player === 'function') ? ui.player(controls, {
    duration: run.T, unit: 'ms', playSeconds: 5, loop: true,
    onFrame: (t) => renderAt(t),
  }) : null;
  if (player) player.seek(0); // start empty; the player then reveals the run
  if (typeof timeline.onResize === 'function') timeline.onResize(() => drawTimeline(curT));
  ui.onTheme(() => renderAt(curT));

  ui.note(notes, `Reward-modulated STDP (RSTDP) splits learning into two stages. First, an <b>eligibility trace</b> <var>c</var> records recent STDP activity and decays with time constant tau_c. Second, the weight only changes when a <b>dopamine</b> signal (driven by a <b>payoff</b>/reward) is present: <var>dw</var> = <var>c</var> · dopamine. Here the reward is delivered only when the post neuron fires inside the adjustable window after the pre spike, so only causally-paired activity is reinforced.`);
  ui.equation(notes, `<var>c</var> += −<var>c</var>/tau_c + <var>stdp_dw</var>; &nbsp; <var>w</var> += <var>c</var> · dopamine; &nbsp; dopamine += −dopamine/tau_dopamine + payoff`);
  ui.note(notes, `All quantities are dimensionless (a.u.); time is in ms. Compare the two weight curves: with a reward (post firing inside the window) the weight moves; without one it barely does, even though the same STDP activity occurred. Push the post delay past the window and the reward stops being delivered. The timeline flashes each reward event as the playhead passes it. ML analogy: reinforcement learning, where a reward (dopamine / TD error) scales a stored eligibility (policy gradient). The analogy breaks because eligibility here is a decaying trace of spike timing, not a full policy gradient.`);
  ui.code(notes, () => `from conex import SimpleRSTDP, Dopamine, Payoff, PreTrace, PostTrace
SimpleRSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c})
Dopamine(tau_dopamine=${state.tau_dopamine})
Payoff(initial_payoff=0.0)  # set network.payoff = ${state.rewardSize} when the post fires within ${state.window} ms of a pre spike`);
  return { destroy() { if (player) player.pause(); } };
}
