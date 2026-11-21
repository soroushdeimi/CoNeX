import {
  DEFAULTS,
  lifStep,
  lifRheobase,
  lifRateAnalytic,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-lif',
  title: 'Leaky integrate-and-fire',
  conex: ['LIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 400;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- controls ---
  const sec = ui.section(controls, 'Model parameters (CoNeX names, a.u.)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'a.u.', min: 1, max: 50, step: 0.5, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'a.u.', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'a.u.', min: 5, max: 40, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'a.u.', min: -10, max: 20, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'a.u.', min: -10, max: 20, step: 0.5, value: P.v_rest });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'a.u.', min: 0.1, max: 5, step: 0.1, value: P.dt });

  const secI = ui.section(controls, 'Input current');
  const cI = ui.slider(secI, { key: 'I', label: 'I (constant)', unit: 'a.u.', min: 0, max: 5, step: 0.05, value: 2.5 });

  // --- figures ---
  const fig = ui.plot(figures, { height: 260, xLabel: 't', yLabel: 'v', xUnit: 'a.u.', yUnit: 'a.u.' });
  const figI = ui.plot(figures, { height: 120, xLabel: 't', yLabel: 'I', xUnit: 'a.u.', yUnit: 'a.u.' });

  // --- notes ---
  ui.note(notes, `The LIF neuron integrates its input current and leaks back to rest. CoNeX updates the
  voltage once per iteration with the Euler step below, then <code>Fire</code> resets any neuron that
  reached the threshold. CoNeX has no refractory period: the reset is instantaneous and the neuron can
  fire again on the very next step.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The <b>rheobase</b> is the smallest constant current that can make the neuron fire. For the
  LIF it is <var>I</var><sub>rheo</sub> = (<var>threshold</var> − <var>v</var><sub>rest</sub>) / <var>R</var>, because that is the current whose
  steady-state voltage <var>v</var><sub>ss</sub> = <var>v</var><sub>rest</sub> + <var>R</var>·<var>I</var> just reaches the threshold. Below it the
  voltage asymptotes to a value under threshold and never fires.`);
  ui.note(notes, `ML analogy: the LIF is the simplest "spike" unit — like a thresholded accumulator. The
  analogy breaks because the leak (<var>tau</var>) makes it a dynamical system with memory, not a static
  threshold, and the firing rate (not the voltage) is what carries information.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(),
      v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get(),
    };
  }

  function run() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], i = [];
    let state = { v: p.v_rest };
    for (let n = 0; n < STEPS; n++) {
      const r = lifStep(state, p, I, p.dt);
      state = { v: r.v };
      t.push(n * p.dt);
      v.push(r.v);
      i.push(I);
    }
    const ir = lifRheobase(p);
    fig.draw(
      [
        { x: t, y: v, label: 'v', color: 'c1', style: 'line' },
      ],
      { hlines: [
        { y: p.threshold, label: 'threshold' },
        { y: ir, label: 'I_rheo (as v_ss)' },
      ] }
    );
    figI.draw([{ x: t, y: i, label: 'I', color: 'c3', style: 'line' }], {});
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}

// Static template with the site default values; the sliders drive the figure.
function codeSnippet() {
  return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${P.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    LIF(R=${P.R}, threshold=${P.threshold}, tau=${P.tau},
        v_reset=${P.v_reset}, v_rest=${P.v_rest}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=2.5)   # constant input current (a.u.)
net.simulate_iterations(${STEPS})`;
}
