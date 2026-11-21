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

  function draw() {
    const p = { ...DEFAULTS };
    const rng = M.mulberry32(state.seed);
    let v = p.v_rest;
    let w = state.w0, preT = 0, postT = 0, spikes = 0;
    let iExc = 0, iInh = 0;
    const decay = Math.exp(-state.dt / state.tau_syn);
    const xs = [], rates = [], ws = [];
    const T = state.steps;
    for (let t = 0; t < T; t++) {
      const preS = rng() < (state.f_pre * state.dt) / 1000 ? 1 : 0;
      const excS = rng() < (state.f_exc * state.dt) / 1000 ? 1 : 0;
      // each spike adds a current that decays with tau_syn (a.u. of pA)
      iExc = iExc * decay + excS * state.I_exc;
      iInh = iInh * decay + preS * w * state.I_inh;
      const I = iExc - iInh; // pA
      const r = lifStep({ v, omega: 0 }, p, I, state.dt);
      v = r.v;
      const postS = r.spike ? 1 : 0;
      spikes += postS;
      preT = M.stepTrace(preT, preS, { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS, { tau_s: state.tau_s, dt: state.dt });
      const dw = M.istdpDw({
        weights: [w], pre_spike: [preS], post_spike: [postS], pre_trace: [preT], post_trace: [postT],
        rho: state.rho, lr: state.lr, is_inhibitory: true, tau_s: state.tau_s,
      })[0];
      w += dw;
      if (t > 0 && t % 100 === 0) {
        xs.push(t * state.dt);
        rates.push(spikes / ((t * state.dt) / 1000));
        ws.push(w);
      }
    }
    const alpha = M.istdpAlpha({ rho: state.rho, tau_s: state.tau_s });
    ratePlot.draw([{ x: xs, y: rates, label: 'post rate', color: 'c1' }], { hlines: [{ y: state.rho, label: 'target rho' }] });
    weightPlot.draw([{ x: xs, y: ws, label: 'weight', color: 'c2' }], { hlines: [{ y: alpha, label: 'alpha' }] });
    state._alpha = alpha;
    state._finalRate = spikes / ((T * state.dt) / 1000);
  }

  const sec = ui.section(controls, 'iSTDP rule');
  ui.slider(sec, { key: 'rho', label: 'rho (target rate)', unit: 'Hz', min: 1, max: 60, step: 1, value: state.rho, onInput: (v) => { state.rho = v; draw(); } });
  ui.slider(sec, { key: 'lr', label: 'lr', unit: 'a.u.', min: 1e-5, max: 0.1, step: 1e-5, value: state.lr, onInput: (v) => { state.lr = v; draw(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; draw(); } });
  ui.slider(sec, { key: 'w0', label: 'w0', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; draw(); } });
  ui.slider(sec, { key: 'f_pre', label: 'pre (inhibitory) rate', unit: 'Hz', min: 1, max: 100, step: 1, value: state.f_pre, onInput: (v) => { state.f_pre = v; draw(); } });
  ui.slider(sec, { key: 'f_exc', label: 'excitatory rate', unit: 'Hz', min: 10, max: 300, step: 5, value: state.f_exc, onInput: (v) => { state.f_exc = v; draw(); } });
  ui.slider(sec, { key: 'I_exc', label: 'excitatory current per spike', unit: 'pA', min: 0, max: 600, step: 10, value: state.I_exc, onInput: (v) => { state.I_exc = v; draw(); } });
  ui.slider(sec, { key: 'I_inh', label: 'inhibitory current per spike', unit: 'pA', min: 0, max: 1000, step: 10, value: state.I_inh, onInput: (v) => { state.I_inh = v; draw(); } });
  ui.slider(sec, { key: 'tau_syn', label: 'synaptic decay', unit: 'ms', min: 1, max: 20, step: 1, value: state.tau_syn, onInput: (v) => { state.tau_syn = v; draw(); } });
  ui.slider(sec, { key: 'steps', label: 'steps', unit: 'ms', min: 5000, max: 100000, step: 5000, value: state.steps, onInput: (v) => { state.steps = v; draw(); } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: state.seed, onInput: (v) => { state.seed = v; draw(); } });

  ui.onTheme(draw);
  draw();

  ui.note(notes, `Symmetric inhibitory STDP (iSTDP) tunes the strength of inhibition so that a postsynaptic neuron settles at a target firing rate <var>rho</var>. Here the post neuron is a real CoNeX LIF driven by Poisson excitatory input and an inhibitory current through the iSTDP weight. When the post neuron fires above target the weight grows to suppress it; below target the weight shrinks. This is a homeostatic mechanism that stabilizes network activity around a set point.`);
  ui.equation(notes, `<var>dw</var> = lr · [ <var>pre_spike</var>·(<var>alpha</var> − <var>post_trace</var>)·sign + <var>pre_trace</var>·<var>post_spike</var> ], &nbsp; <var>alpha</var> = 2·<var>rho</var>·tau_s / 1000`);
  ui.note(notes, `Input model: each excitatory spike adds a current of I_exc (pA) and each inhibitory (pre) spike adds w·I_inh (pA); both decay with time constant tau_syn (ms). The LIF is in mV/ms/pA (neurons.js, Gerstner et al. Table 6.1: R=0.5 GOhm, v_rest=−70 mV, threshold=0 mV). The values are chosen so the post rate sits in the 5-30 Hz range and the weight can move it toward rho; they are not from a specific dataset. The top plot shows the post rate approaching the target rho; the bottom plot shows the weight trace.`);
  ui.note(notes, `ML analogy: a gain controller that adjusts a feedback weight to hold an output at a reference. The analogy breaks because the "error" is a spike trace versus a target, not a scalar loss, and the update is driven by spike timing.`);
  ui.code(notes, () => `from conex import SimpleiSTDP, PreTrace, PostTrace, LIF
SimpleiSTDP(rho=${state.rho}, lr=${state.lr}, is_inhibitory=True)
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})
LIF(R=0.5, threshold=0.0, tau=20.0, v_reset=-55.0, v_rest=-70.0)
# alpha = 2 * rho * tau_s / 1000 = ${state._alpha.toPrecision(4)}`);
  return { destroy() {} };
}
