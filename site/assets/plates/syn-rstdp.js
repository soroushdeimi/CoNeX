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

  function run(rewardOn) {
    const T = state.steps;
    let w = state.w0, c = [0], dop = 0, preT = 0, postT = 0;
    let lastPre = -1e9;
    const xs = [], ws = [], cs = [], dops = [];
    for (let t = 0; t < T; t++) {
      const preS = t % state.prePeriod === 0 ? 1 : 0;
      if (preS) lastPre = t;
      const postS = t >= state.offset && (t - state.offset) % state.prePeriod === 0 ? 1 : 0;
      // reward only when the post fires inside `window` steps after a pre spike
      const payoff = rewardOn && postS && (t - lastPre) <= state.window ? state.rewardSize : 0;
      dop = M.dopamineStep({ dopamine_concentration: dop, tau_dopamine: state.tau_dopamine, payoff });
      preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.stdpDw({ weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus });
      const step = M.rstdpStep({ c, dw, tau_c: state.tau_c, dopamine_concentration: dop });
      c = step.c;
      w += step.dwApplied[0];
      xs.push(t); ws.push(w); cs.push(c[0]); dops.push(dop);
    }
    return { xs, ws, cs, dops };
  }

  function draw() {
    const withR = run(true);
    const noR = run(false);
    weightPlot.draw([
      { x: withR.xs, y: withR.ws, label: 'weight (reward in window)', color: 'c1' },
      { x: noR.xs, y: noR.ws, label: 'weight (no reward)', color: 'c2' },
    ]);
    dopPlot.draw([{ x: withR.xs, y: withR.dops, label: 'dopamine', color: 'c4' }], { yRange: [0, Math.max(0.1, ...withR.dops)] });
    eligPlot.draw([{ x: withR.xs, y: withR.cs, label: 'eligibility c', color: 'c3' }], { yRange: [0, Math.max(0.01, ...withR.cs)] });
  }

  const sec = ui.section(controls, 'RSTDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; draw(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; draw(); } });
  ui.slider(sec, { key: 'tau_c', label: 'tau_c (eligibility)', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_c, onInput: (v) => { state.tau_c = v; draw(); } });
  ui.slider(sec, { key: 'tau_dopamine', label: 'tau_dopamine', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_dopamine, onInput: (v) => { state.tau_dopamine = v; draw(); } });
  ui.slider(sec, { key: 'offset', label: 'post delay after pre', unit: 'ms', min: 0, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; draw(); } });
  ui.slider(sec, { key: 'window', label: 'reward window', unit: 'ms', min: 0, max: 10, step: 1, value: state.window, onInput: (v) => { state.window = v; draw(); } });
  ui.slider(sec, { key: 'rewardSize', label: 'reward size', unit: 'a.u.', min: 0, max: 3, step: 0.1, value: state.rewardSize, onInput: (v) => { state.rewardSize = v; draw(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'ms', min: 20, max: 200, step: 10, value: state.steps, onInput: (v) => { state.steps = v; draw(); } });

  ui.onTheme(draw);
  draw();

  ui.note(notes, `Reward-modulated STDP (RSTDP) splits learning into two stages. First, an <b>eligibility trace</b> <var>c</var> records recent STDP activity and decays with time constant tau_c. Second, the weight only changes when a <b>dopamine</b> signal (driven by a <b>payoff</b>/reward) is present: <var>dw</var> = <var>c</var> · dopamine. Here the reward is delivered only when the post neuron fires inside the adjustable window after the pre spike, so only causally-paired activity is reinforced.`);
  ui.equation(notes, `<var>c</var> += −<var>c</var>/tau_c + <var>stdp_dw</var>; &nbsp; <var>w</var> += <var>c</var> · dopamine; &nbsp; dopamine += −dopamine/tau_dopamine + payoff`);
  ui.note(notes, `All quantities are dimensionless (a.u.); time is in ms. Compare the two weight curves: with a reward (post firing inside the window) the weight moves; without one it barely does, even though the same STDP activity occurred. Push the post delay past the window and the reward stops being delivered. ML analogy: reinforcement learning, where a reward (dopamine / TD error) scales a stored eligibility (policy gradient). The analogy breaks because eligibility here is a decaying trace of spike timing, not a full policy gradient.`);
  ui.code(notes, () => `from conex import SimpleRSTDP, Dopamine, Payoff, PreTrace, PostTrace
SimpleRSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c})
Dopamine(tau_dopamine=${state.tau_dopamine})
Payoff(initial_payoff=0.0)  # set network.payoff = ${state.rewardSize} when the post fires within ${state.window} ms of a pre spike`);
  return { destroy() {} };
}
