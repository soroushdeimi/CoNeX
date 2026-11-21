import { DEFAULTS, elifStep } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-elif',
  title: 'Exponential integrate-and-fire',
  conex: ['ELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 500;

// dv/dt for the ELIF at a given v and constant current I (mV per ms).
function dvdt(v, p, I) {
  const Fu = p.v_rest - v + p.delta * Math.exp((v - p.theta_rh) / p.delta);
  return (Fu + p.R * I) / p.tau;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -20, max: 20, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'mV', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'mV', min: -70, max: -30, step: 0.5, value: P.theta_rh });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  const figPhase = ui.plot(figures, { height: 240, xLabel: 'v', yLabel: 'dv/dt', xUnit: 'mV', yUnit: 'mV/ms' });
  const figV = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });

  ui.note(notes, `The ELIF adds an exponential spike-initiation term to the LIF leak. As the voltage
  approaches the boosting threshold <var>theta</var><sub>rh</sub>, the term <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>)
  grows sharply, so the neuron accelerates toward threshold on its own. <var>delta</var> sets how sharp that
  take-off is; <var>theta</var><sub>rh</sub> sets where it starts. Units: v in mV, t in ms, I in pA, R in GOhm. The
  spike-detection threshold (0 mV here) is a simulation choice — Gerstner gives no value for it.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The phase line plots d<var>v</var>/d<var>t</var> against <var>v</var> for the current input. Where the curve
  crosses zero is a fixed point: if it is below threshold the neuron settles there (no firing); if the
  input is strong enough that no stable fixed point sits below threshold, the voltage runs away and fires.
  The vertical axis is clipped to a readable window; above <var>theta</var><sub>rh</sub> the exponential term makes
  d<var>v</var>/d<var>t</var> grow without bound, so the curve leaves the top of the window there.
  In the voltage trace the spike peaks look like they only reach about −25 mV even though the
  detection threshold is 0 mV: at dt=0.1 ms the upstroke is a single Euler step, so the step that
  crosses threshold is reset before it is recorded and the plotted peak is the last value below
  threshold, not the true peak.
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
    // Clip the y window to the interesting part: the exponential term blows up
    // above ~theta_rh, so a fixed window keeps the fixed points readable.
    const yLo = -2, yHi = 10; // mV/ms
    figPhase.draw(
      [{ x: xs, y: ys, label: 'dv/dt', color: 'c1', style: 'line' }],
      { yRange: [yLo, yHi],
        hlines: [{ y: 0, label: 'dv/dt = 0' }],
        vlines: [
          ...fixed.map((f) => ({ x: f, label: 'fixed point' })),
          { x: p.theta_rh, label: 'theta_rh' },
        ] }
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

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import ELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    ELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
         v_reset=${p.v_reset}, v_rest=${p.v_rest},
         delta=${p.delta}, theta_rh=${p.theta_rh}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
