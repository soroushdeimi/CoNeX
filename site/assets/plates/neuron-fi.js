import { DEFAULTS, simulate, lifRateAnalytic } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-fi',
  title: 'f-I curves',
  conex: ['LIF', 'ELIF', 'AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 2000;
const CURRENTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Shared parameters (a.u.)');
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

  const fig = ui.plot(figures, { height: 300, xLabel: 'I', yLabel: 'f', xUnit: 'a.u.', yUnit: 'Hz (1/a.u.)' });

  ui.note(notes, `The f-I curve plots the steady firing rate <var>f</var> against the input current <var>I</var>. Each
  point is measured by simulating that current for a fixed number of steps and counting spikes, so the
  curves are the real CoNeX dynamics, not a formula. The dashed line is the analytic LIF rate
  <var>f</var> = 1 / (<var>tau</var>·ln((<var>v</var><sub>ss</sub>−<var>v</var><sub>reset</sub>)/(<var>v</var><sub>ss</sub>−<var>threshold</var>))), <var>v</var><sub>ss</sub> = <var>v</var><sub>rest</sub> + <var>R</var>·<var>I</var>.`);
  ui.note(notes, `ELIF fires at lower currents than LIF because the exponential term pushes it over
  threshold; AELIF fires least of all because the adaptation current <var>w</var> subtracts from the input.
  The <b>dt slider</b> shows the Euler discretization: a large <var>dt</var> changes the measured rate and can
  even change whether a neuron fires at all, because the update is a first-order approximation.
  ML analogy: the f-I curve is the neuron's "activation function" relating input to output rate; the
  analogy breaks because it is a dynamical, history-dependent rate, not a static function of the input.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: cDt.get(),
    };
  }

  function rate(model, p, I, dt) {
    const r = simulate(model, p, I, STEPS, dt);
    return r.spikes.reduce((a, b) => a + (b ? 1 : 0), 0) / (STEPS * dt);
  }

  function run() {
    const p = params();
    const dt = p.dt;
    const series = [];
    const colors = { lif: 'c1', elif: 'c2', aelif: 'c4' };
    for (const model of ['lif', 'elif', 'aelif']) {
      series.push({
        x: CURRENTS.slice(),
        y: CURRENTS.map((c) => rate(model, p, c, dt)),
        label: model.toUpperCase(),
        color: colors[model],
        style: 'line',
      });
    }
    // Analytic LIF rate as a dashed line.
    series.push({
      x: CURRENTS.slice(),
      y: CURRENTS.map((c) => lifRateAnalytic(p, c)),
      label: 'LIF analytic',
      color: 'c1',
      style: 'dots',
    });
    fig.draw(series, {});
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}

function codeSnippet() {
  return `# f-I: for each current I, run the neuron and count spikes over ${STEPS} steps.
from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF, AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

def fi(behavior, currents, steps=${STEPS}, dt=${P.dt}):
    rates = []
    for I in currents:
        net = Network(behavior=[TimeResolution(dt=dt)])
        pop = NeuronGroup(net=net, size=1, behavior=[behavior, Fire()])
        net.initialize(); pop.I = pop.vector(mode=I)
        net.simulate_iterations(steps)
        rates.append(pop.spikes.sum().item() / (steps * dt))
    return rates

currents = ${JSON.stringify(CURRENTS)}
lif   = fi(LIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau}, v_reset=${P.v_reset}, v_rest=${P.v_rest}), currents)
elif  = fi(ELIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau}, v_reset=${P.v_reset}, v_rest=${P.v_rest},
                delta=${P.delta}, theta_rh=${P.theta_rh}), currents)
aelif = fi(AELIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau}, v_reset=${P.v_reset}, v_rest=${P.v_rest},
                 delta=${P.delta}, theta_rh=${P.theta_rh}, alpha=${P.alpha}, beta=${P.beta}, w_tau=${P.w_tau}), currents)`;
}
