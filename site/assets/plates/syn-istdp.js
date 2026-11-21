// Plate: Inhibitory STDP (SimpleiSTDP) drives the postsynaptic rate toward a target.
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'syn-istdp',
  title: 'Inhibitory STDP',
  conex: ['SimpleiSTDP'],
  source: 'conex/behaviors/synapses/learning.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = { rho: 0.5, lr: 1e-5, tau_s: 10, dt: 1, w0: 0.5, steps: 20000, preRate: 0.3, postBase: 0.3, gain: 3.0 };

  const weightPlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'weight', yUnit: 'a.u.' });
  const ratePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'post trace', yUnit: 'a.u.' });

  function draw() {
    const T = state.steps;
    const rand = M.mulberry32(123);
    let w = state.w0, preT = 0, postT = 0;
    const xs = [], ws = [], rates = [];
    for (let t = 0; t < T; t++) {
      const preS = rand() < state.preRate ? 1 : 0;
      // simplified closed loop: as the inhibitory weight grows above w0, the post rate falls
      const postProb = Math.max(0, Math.min(1, state.postBase - (w - state.w0) * state.gain));
      const postS = rand() < postProb ? 1 : 0;
      preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.istdpDw({ weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT], rho: state.rho, lr: state.lr, is_inhibitory: true, tau_s: state.tau_s })[0];
      w += dw;
      xs.push(t); ws.push(w); rates.push(postT);
    }
    const alpha = M.istdpAlpha({ rho: state.rho, tau_s: state.tau_s });
    weightPlot.draw([{ x: xs, y: ws, label: 'inhibitory weight', color: 'c2' }], { hlines: [{ y: alpha, label: 'alpha (target trace)' }] });
    ratePlot.draw([
      { x: xs, y: rates, label: 'post_trace (rate proxy)', color: 'c1' },
    ], { hlines: [{ y: alpha, label: 'alpha = 2·rho·tau_s/1000' }] });
    state._alpha = alpha;
  }

  const sec = ui.section(controls, 'iSTDP rule');
  ui.slider(sec, { key: 'rho', label: 'rho (target rate)', unit: 'Hz', min: 0.01, max: 1.0, step: 0.01, value: state.rho, onInput: (v) => { state.rho = v; draw(); } });
  ui.slider(sec, { key: 'lr', label: 'lr', unit: 'a.u.', min: 1e-6, max: 1e-3, step: 1e-6, value: state.lr, onInput: (v) => { state.lr = v; draw(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; draw(); } });
  ui.slider(sec, { key: 'w0', label: 'w0', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; draw(); } });
  ui.slider(sec, { key: 'preRate', label: 'pre rate', unit: 'Hz', min: 0.05, max: 0.6, step: 0.05, value: state.preRate, onInput: (v) => { state.preRate = v; draw(); } });
  ui.slider(sec, { key: 'postBase', label: 'post base rate', unit: 'Hz', min: 0.05, max: 0.6, step: 0.05, value: state.postBase, onInput: (v) => { state.postBase = v; draw(); } });
  ui.slider(sec, { key: 'gain', label: 'inhibitory gain', unit: 'a.u.', min: 0, max: 10, step: 0.5, value: state.gain, onInput: (v) => { state.gain = v; draw(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'ms', min: 1000, max: 50000, step: 1000, value: state.steps, onInput: (v) => { state.steps = v; draw(); } });

  ui.onTheme(draw);
  draw();

  ui.note(notes, `Symmetric inhibitory STDP (iSTDP) tunes the strength of inhibition so that the postsynaptic neuron settles at a target firing rate <var>rho</var>. When the post neuron fires too much, the inhibitory weight grows to suppress it; when it fires too little, the weight shrinks. This is a homeostatic mechanism: it stabilizes network activity around a set point. Here the post rate is a simplified closed loop (post probability falls as the inhibitory weight rises) so you can watch the weight push the post trace toward the target; CoNeX itself only implements the weight update, not the downstream neuron.`);
  ui.equation(notes, `<var>dw</var> = lr · [ <var>pre_spike</var>·(<var>alpha</var> − <var>post_trace</var>)·sign + <var>pre_trace</var>·<var>post_spike</var> ], &nbsp; <var>alpha</var> = 2·<var>rho</var>·tau_s / 1000`);
  ui.note(notes, `tau_s is in ms (CoNeX assumes this); the weight is dimensionless (a.u.). The sign is negative for an inhibitory synapse (is_inhibitory=True). ML analogy: this is like a gain controller that adjusts a feedback weight to hold an output at a reference value. The analogy breaks because the "error" here is the difference between a spike trace and a target, not a scalar loss, and the update is driven by spike timing.`);
  ui.code(notes, () => `from conex import SimpleiSTDP, PreTrace, PostTrace
SimpleiSTDP(rho=${state.rho}, lr=${state.lr}, is_inhibitory=True)
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})
# alpha = 2 * rho * tau_s / 1000 = ${state._alpha}`);
  return { destroy() {} };
}
