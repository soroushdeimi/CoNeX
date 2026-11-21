// Plate: Reward-modulated STDP (SimpleRSTDP) with Dopamine and Payoff.
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
    tau_dopamine: 10, steps: 40, rewardAt: 15, rewardSize: 1.0,
  };

  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'weight', yUnit: 'a.u.' });
  const modPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'dopamine / eligibility', yUnit: 'a.u.' });

  function run(rewardOn) {
    const T = state.steps;
    const rand = M.mulberry32(7);
    let w = state.w0, c = 0, dop = 0, preT = 0, postT = 0;
    const xs = [], ws = [], cs = [], dops = [];
    for (let t = 0; t < T; t++) {
      const payoff = rewardOn && t === state.rewardAt ? state.rewardSize : 0;
      dop = M.dopamineStep({ dopamine_concentration: dop, tau_dopamine: state.tau_dopamine, payoff });
      const preS = t % 5 === 0 ? 1 : 0;
      const postS = t % 5 === 2 ? 1 : 0;
      preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.stdpDw({ weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus })[0];
      const step = M.rstdpStep({ c, dw, tau_c: state.tau_c, dopamine_concentration: dop });
      c = step.c;
      w += step.dwApplied;
      xs.push(t); ws.push(w); cs.push(c); dops.push(dop);
    }
    return { xs, ws, cs, dops };
  }

  function draw() {
    const withR = run(true);
    const noR = run(false);
    weightPlot.draw([
      { x: withR.xs, y: withR.ws, label: 'weight (with reward)', color: 'c1' },
      { x: noR.xs, y: noR.ws, label: 'weight (no reward)', color: 'c2' },
    ], { vlines: [{ x: state.rewardAt, label: 'reward' }] });
    modPlot.draw([
      { x: withR.xs, y: withR.dops, label: 'dopamine', color: 'c4' },
      { x: withR.xs, y: withR.cs, label: 'eligibility c', color: 'c3' },
    ], { vlines: [{ x: state.rewardAt, label: 'reward' }] });
  }

  const sec = ui.section(controls, 'RSTDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; draw(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; draw(); } });
  ui.slider(sec, { key: 'tau_c', label: 'tau_c (eligibility)', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_c, onInput: (v) => { state.tau_c = v; draw(); } });
  ui.slider(sec, { key: 'tau_dopamine', label: 'tau_dopamine', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_dopamine, onInput: (v) => { state.tau_dopamine = v; draw(); } });
  ui.slider(sec, { key: 'rewardAt', label: 'reward time', unit: 'ms', min: 0, max: 39, step: 1, value: state.rewardAt, onInput: (v) => { state.rewardAt = v; draw(); } });
  ui.slider(sec, { key: 'rewardSize', label: 'reward size', unit: 'a.u.', min: 0, max: 3, step: 0.1, value: state.rewardSize, onInput: (v) => { state.rewardSize = v; draw(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'ms', min: 10, max: 100, step: 5, value: state.steps, onInput: (v) => { state.steps = v; draw(); } });

  ui.onTheme(draw);
  draw();

  ui.note(notes, `Reward-modulated STDP (RSTDP) splits learning into two stages. First, an <b>eligibility trace</b> <var>c</var> records recent STDP activity and decays with time constant tau_c. Second, the weight only changes when a <b>dopamine</b> signal (driven by a <b>payoff</b>/reward) is present: <var>dw</var> = <var>c</var> · dopamine. This is the standard model of how a reward gates which synapses are strengthened.`);
  ui.equation(notes, `<var>c</var> += −<var>c</var>/tau_c + <var>stdp_dw</var>; &nbsp; <var>w</var> += <var>c</var> · dopamine; &nbsp; dopamine += −dopamine/tau_dopamine + payoff`);
  ui.note(notes, `All quantities are dimensionless (a.u.); time is in ms. Compare the two weight curves: with a reward the weight moves, without one it barely does, even though the same STDP activity occurred. ML analogy: this is like reinforcement learning, where a reward signal (the dopamine / TD error) scales a stored eligibility (the policy gradient). The analogy breaks because eligibility here is a decaying trace of spike timing, not a full policy gradient.`);
  ui.code(notes, () => `from conex import SimpleRSTDP, Dopamine, Payoff, PreTrace, PostTrace
SimpleRSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c})
Dopamine(tau_dopamine=${state.tau_dopamine})
Payoff(initial_payoff=0.0)  # set network.payoff = ${state.rewardSize} at t=${state.rewardAt}`);
  return { destroy() {} };
}
