import { DEFAULTS, lifUpdate, kwta, inherentNoise } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-kwta',
  title: 'k-winners-take-all',
  conex: ['KWTA', 'InherentNoise'],
  source: 'conex/behaviors/neurons/specs.py',
};

const P = { ...DEFAULTS };
const SIZE = 8;
const STEPS = 200;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Neuron (LIF) parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });

  const secIn = ui.section(controls, 'Input and noise (seeded)');
  const cSeed = ui.number(secIn, { key: 'seed', label: 'seed', unit: '', value: 0 });
  const cI = ui.slider(secIn, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });
  const cScale = ui.slider(secIn, { key: 'scale', label: 'noise scale', unit: 'mV', min: 0, max: 20, step: 0.5, value: 5.0 });
  const cOffset = ui.slider(secIn, { key: 'offset', label: 'noise offset', unit: 'mV', min: -10, max: 10, step: 0.5, value: 0.0 });

  const secK = ui.section(controls, 'KWTA');
  const cK = ui.slider(secK, { key: 'k', label: 'k (winners)', unit: '', min: 1, max: SIZE, step: 1, value: 2 });

  const figBefore = ui.raster(figures, { height: 200, xLabel: 't', xUnit: 'ms' });
  const figAfter = ui.raster(figures, { height: 200, xLabel: 't', xUnit: 'ms' });

  ui.note(notes, `KWTA keeps only the <var>k</var> strongest neurons active. In CoNeX, <code>KWTA.forward</code>
  checks which neurons are above threshold; if more than <var>k</var> are, it keeps the top-<var>k</var> by voltage and
  resets every other spiking neuron to <var>v_reset</var>. If at most <var>k</var> are above threshold, nothing changes.
  <code>InherentNoise</code> adds a uniform random voltage offset (mV) each step so the group's activity varies.
  Units: v in mV, t in ms, I in pA, R in GOhm.`);
  ui.note(notes, `The two rasters use the <b>same seeded noise</b>: the top is the group with no KWTA
  (every neuron that crosses threshold fires), the bottom is the same group with KWTA applied, so the
  difference is exactly the k-winners inhibition. Raising <var>k</var> lets more neurons win; lowering it
  sharpens the competition. ML analogy: KWTA is a hard top-<var>k</var> sparsification, like keeping only the
  <var>k</var> largest activations; the analogy breaks because here the "activations" are membrane voltages
  with dynamics, not a static feature vector.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return { R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get() };
  }

  // Run the group for STEPS steps. Both runs apply the same seeded noise; the only
  // difference is whether KWTA resets the losers before the spike decision.
  function run(useKwta) {
    const p = params();
    const rng = ui.rng(cSeed.get());
    const I = cI.get();
    const scale = cScale.get();
    const offset = cOffset.get();
    const k = cK.get();
    let vs = new Array(SIZE).fill(p.v_rest);
    const spikes = []; // [t, index]
    for (let it = 1; it <= STEPS; it++) {
      const t = it * p.dt;
      vs = vs.map((v) => lifUpdate({ v }, p, I, p.dt));
      vs = inherentNoise(vs, rng, scale, offset);
      if (useKwta) vs = kwta(vs, p.threshold, p.v_reset, k);
      for (let j = 0; j < SIZE; j++) {
        if (vs[j] >= p.threshold) { spikes.push([t, j]); vs[j] = p.v_reset; }
      }
    }
    return spikes;
  }

  function draw() {
    const tMax = STEPS * params().dt;
    figBefore.draw(run(false), SIZE, tMax);
    figAfter.draw(run(true), SIZE, tMax);
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup, NeuronDimension
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire, KWTA, InherentNoise
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=1, width=${SIZE}), behavior=[
    LIF(R=${p.R}, threshold=${cTh.get()}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    InherentNoise(mode="rand", scale=${cScale.get()}, offset=${cOffset.get()}),
    KWTA(k=${cK.get()}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  const all = [cR, cTau, cTh, cReset, cRest, cDt, cSeed, cI, cScale, cOffset, cK];
  for (const c of all) c.onInput(draw);
  ui.onTheme(draw);
  draw();

  return { destroy() {} };
}
