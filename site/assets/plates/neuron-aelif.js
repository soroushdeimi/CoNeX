import { DEFAULTS, TABLE61, aelifStep, spikeTimes, firingStats } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-aelif',
  title: 'Adaptive exponential integrate-and-fire',
  conex: ['AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 5000;
// Rows whose named pattern is NOT reproduced by CoNeX's AELIF at dt=0.1 ms.
const NOT_REPRODUCED = new Set(['Bursting', 'Irregular']);

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const secP = ui.section(controls, 'Firing pattern (Gerstner Table 6.1)');
  const cPattern = ui.select(secP, {
    key: 'pattern', label: 'pattern',
    options: Object.keys(TABLE61).map((k) => ({ value: k, label: k })),
    value: 'Tonic',
  });

  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -20, max: 20, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -70, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'mV', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'mV', min: -70, max: -30, step: 0.5, value: P.theta_rh });
  const cAlpha = ui.slider(sec, { key: 'alpha', label: 'alpha', unit: 'nS', min: -2, max: 2, step: 0.05, value: P.alpha });
  const cBeta = ui.slider(sec, { key: 'beta', label: 'beta', unit: 'pA', min: 0, max: 100, step: 1, value: P.beta });
  const cWTau = ui.slider(sec, { key: 'w_tau', label: 'w_tau', unit: 'ms', min: 5, max: 300, step: 5, value: P.w_tau });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  const figV = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figW = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'w (omega)', xUnit: 'ms', yUnit: 'pA' });
  const status = document.createElement('p');
  status.style.fontStyle = 'italic';
  figures.appendChild(status);

  ui.note(notes, `The AELIF adds an adaptation current <var>w</var> (called <var>omega</var> in CoNeX) to the ELIF.
  It has two drivers: a subthreshold term <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) that builds up while the neuron is
  active, and a spike-triggered kick <var>beta</var>·<var>w</var><sub>tau</sub> on every spike. The current <var>R</var>·<var>w</var> is subtracted
  from the input, so as <var>w</var> grows the neuron fires more slowly — this is <b>adaptation</b>.`);
  ui.equation(notes, `<var>tau</var><sub>w</sub>·d<var>w</var>/d<var>t</var> = <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) − <var>w</var> + <var>beta</var>·<var>tau</var><sub>w</sub>·spikes`);
  ui.note(notes, `The "Firing pattern" select loads the seven rows of Gerstner et al. Table 6.1 (R=0.5 GOhm,
  v<sub>rest</sub>=−70 mV, <var>theta</var><sub>rh</sub>=−50 mV, <var>delta</var>=2 mV, threshold=0 mV, dt=0.1 ms). The spike-detection
  threshold of 0 mV is a simulation choice — the book gives no value for it. With the CoNeX equations
  at dt=0.1 ms, Tonic, Adapting, Initial burst, Transient and Delayed reproduce their named patterns;
  <b>Bursting and Irregular do not</b> (both fire nearly regularly), so they are labelled "not reproduced
  at this dt". ML analogy: <var>w</var> is a short-term "fatigue" state like a running average of recent
  activity; the analogy breaks because it is a continuous dynamical variable, not a discrete counter.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: cDt.get(),
    };
  }

  function applyPattern() {
    const row = TABLE61[cPattern.get()];
    cTau.set(row.tau); cAlpha.set(row.alpha); cWTau.set(row.w_tau);
    cBeta.set(row.beta); cReset.set(row.v_reset); cI.set(row.I);
    run();
  }

  function run() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], w = [], spikes = [];
    let state = { v: p.v_rest, omega: 0.0 };
    for (let n = 0; n < STEPS; n++) {
      const r = aelifStep(state, p, I, p.dt);
      state = { v: r.v, omega: r.omega };
      t.push(n * p.dt); v.push(r.v); w.push(r.omega); spikes.push(r.spike);
    }
    figV.draw([{ x: t, y: v, label: 'v', color: 'c1', style: 'line' }],
      { hlines: [{ y: p.threshold, label: 'threshold' }] });
    figW.draw([{ x: t, y: w, label: 'w', color: 'c4', style: 'line' }], {});

    const s = firingStats(spikeTimes(spikes, p.dt));
    const name = cPattern.get();
    const reproduced = !NOT_REPRODUCED.has(name);
    const first = s.first_t == null ? '—' : s.first_t.toPrecision(4);
    status.textContent = `${name}: ${s.n} spikes in ${(STEPS * p.dt).toPrecision(4)} ms, CV=${s.cv?.toPrecision(3) ?? '—'}, ` +
      `first spike ${first} ms — ${reproduced ? 'pattern reproduced' : 'not reproduced at this dt'}`;
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    AELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
          v_reset=${p.v_reset}, v_rest=${p.v_rest},
          delta=${p.delta}, theta_rh=${p.theta_rh},
          alpha=${p.alpha}, beta=${p.beta}, w_tau=${p.w_tau}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  cPattern.onInput(applyPattern);
  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
