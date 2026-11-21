import { DEFAULTS, aelifStep } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-aelif',
  title: 'Adaptive exponential integrate-and-fire',
  conex: ['AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 600;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Model parameters (CoNeX names, a.u.)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'a.u.', min: 1, max: 50, step: 0.5, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'a.u.', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'a.u.', min: 5, max: 40, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'a.u.', min: -10, max: 20, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'a.u.', min: -10, max: 20, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'a.u.', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'a.u.', min: 0, max: 30, step: 0.5, value: P.theta_rh });
  const cAlpha = ui.slider(sec, { key: 'alpha', label: 'alpha', unit: 'a.u.', min: 0, max: 1, step: 0.01, value: P.alpha });
  const cBeta = ui.slider(sec, { key: 'beta', label: 'beta', unit: 'a.u.', min: 0, max: 5, step: 0.1, value: P.beta });
  const cWTau = ui.slider(sec, { key: 'w_tau', label: 'w_tau', unit: 'a.u.', min: 5, max: 300, step: 5, value: P.w_tau });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'a.u.', min: 0.1, max: 5, step: 0.1, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'a.u.', min: 0, max: 6, step: 0.05, value: 4.0 });

  const figV = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'a.u.', yUnit: 'a.u.' });
  const figW = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'w (omega)', xUnit: 'a.u.', yUnit: 'a.u.' });

  ui.note(notes, `The AELIF adds an adaptation current <var>w</var> (called <var>omega</var> in CoNeX) to the ELIF.
  It has two drivers: a subthreshold term <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) that builds up while the neuron is
  active, and a spike-triggered kick <var>beta</var>·<var>w</var><sub>tau</sub> on every spike. The current <var>R</var>·<var>w</var> is subtracted
  from the input, so as <var>w</var> grows the neuron fires more slowly — this is <b>adaptation</b>.`);
  ui.equation(notes, `<var>tau</var><sub>a</sub>·d<var>w</var>/d<var>t</var> = <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) − <var>w</var> + <var>beta</var>·<var>tau</var><sub>a</sub>·spikes`);
  ui.note(notes, `Raising <var>beta</var> (spike-triggered) or <var>alpha</var> (subthreshold) strengthens adaptation and
  slows the firing rate; a short <var>w</var><sub>tau</sub> lets <var>w</var> decay fast, so the neuron recovers and fires faster.
  With the CoNeX equations and these parameters the neuron shows sustained adaptation (the rate slows
  as <var>w</var> rises) but does not burst — we do not claim bursting here. ML analogy: <var>w</var> is a short-term
  "fatigue" state like a running average of recent activity; the analogy breaks because it is a
  continuous dynamical variable, not a discrete counter.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: cDt.get(),
    };
  }

  function run() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], w = [];
    let state = { v: p.v_rest, omega: 0.0 };
    for (let n = 0; n < STEPS; n++) {
      const r = aelifStep(state, p, I, p.dt);
      state = { v: r.v, omega: r.omega };
      t.push(n * p.dt); v.push(r.v); w.push(r.omega);
    }
    figV.draw([{ x: t, y: v, label: 'v', color: 'c1', style: 'line' }],
      { hlines: [{ y: p.threshold, label: 'threshold' }] });
    figW.draw([{ x: t, y: w, label: 'w', color: 'c4', style: 'line' }], {});
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}

function codeSnippet() {
  return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${P.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    AELIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau},
          v_reset=${P.v_reset}, v_rest=${P.v_rest},
          delta=${P.delta}, theta_rh=${P.theta_rh},
          alpha=${P.alpha}, beta=${P.beta}, w_tau=${P.w_tau}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=4.0)   # constant input current (a.u.)
net.simulate_iterations(${STEPS})`;
}
