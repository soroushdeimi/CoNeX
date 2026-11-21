import { DEFAULTS, elifStep } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-elif',
  title: 'Exponential integrate-and-fire',
  conex: ['ELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 400;

// dv/dt for the ELIF at a given v and constant current I (a.u. per a.u. of time).
function dvdt(v, p, I) {
  const Fu = p.v_rest - v + p.delta * Math.exp((v - p.theta_rh) / p.delta);
  return (Fu + p.R * I) / p.tau;
}

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
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'a.u.', min: 0.1, max: 5, step: 0.1, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'a.u.', min: 0, max: 5, step: 0.05, value: 1.0 });

  const figPhase = ui.plot(figures, { height: 240, xLabel: 'v', yLabel: 'dv/dt', xUnit: 'a.u.', yUnit: 'a.u./a.u.' });
  const figV = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'v', xUnit: 'a.u.', yUnit: 'a.u.' });

  ui.note(notes, `The ELIF adds an exponential spike-initiation term to the LIF leak. As the voltage
  approaches the boosting threshold <var>theta</var><sub>rh</sub>, the term <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>)
  grows sharply, so the neuron accelerates toward threshold on its own. <var>delta</var> sets how sharp that
  take-off is; <var>theta</var><sub>rh</sub> sets where it starts.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The phase line plots d<var>v</var>/d<var>t</var> against <var>v</var> for the current input. Where the curve
  crosses zero is a fixed point: if it is below threshold the neuron settles there (no firing); if the
  input is strong enough that no stable fixed point sits below threshold, the voltage runs away and fires.
  ML analogy: the exponential term is like a soft, self-reinforcing "excitation" — the analogy breaks
  because it is a deterministic dynamical term, not a learned weight.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(), dt: cDt.get(),
    };
  }

  function run() {
    const p = params();
    const I = cI.get();

    // Phase line: dv/dt over a range of v, mark zero crossings (fixed points).
    const vMin = Math.min(p.v_rest, p.v_reset) - 5;
    const vMax = p.threshold + 5;
    const N = 200;
    const xs = [], ys = [];
    let prev = null;
    const fixed = [];
    for (let k = 0; k <= N; k++) {
      const v = vMin + (vMax - vMin) * k / N;
      const d = dvdt(v, p, I);
      xs.push(v); ys.push(d);
      if (prev !== null && (prev.d <= 0) !== (d <= 0)) {
        const t = prev.d / (prev.d - d);
        fixed.push(prev.v + t * (v - prev.v));
      }
      prev = { v, d };
    }
    figPhase.draw(
      [{ x: xs, y: ys, label: 'dv/dt', color: 'c1', style: 'line' }],
      { hlines: [{ y: 0, label: 'dv/dt = 0' }],
        vlines: fixed.map((f) => ({ x: f, label: 'fixed point' })) }
    );

    // Voltage trace.
    const t = [], v = [];
    let state = { v: p.v_rest };
    for (let n = 0; n < STEPS; n++) {
      const r = elifStep(state, p, I, p.dt);
      state = { v: r.v };
      t.push(n * p.dt); v.push(r.v);
    }
    figV.draw([{ x: t, y: v, label: 'v', color: 'c2', style: 'line' }],
      { hlines: [{ y: p.threshold, label: 'threshold' }] });
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}

function codeSnippet() {
  return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import ELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${P.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    ELIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau},
         v_reset=${P.v_reset}, v_rest=${P.v_rest},
         delta=${P.delta}, theta_rh=${P.theta_rh}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=1.0)   # constant input current (a.u.)
net.simulate_iterations(${STEPS})`;
}
