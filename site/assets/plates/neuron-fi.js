import { DEFAULTS, simulate, lifRateAnalytic } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-fi',
  title: 'f-I curves',
  conex: ['LIF', 'ELIF', 'AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 5000;
const CURRENTS = [20, 40, 60, 80, 100, 120]; // pA
const TRACE_STEPS = 500; // steps for the small voltage trace at a given I

// Use ui.player when the shell provides it; otherwise fall back to a stub that
// reveals the whole sweep once, so figures are never empty in the test mock.
function makePlayer(ui, parent, onFrame) {
  if (typeof ui.player === 'function') {
    return ui.player(parent, { duration: 0, unit: 'pA', playSeconds: 5, loop: true, onFrame });
  }
  let t = 0;
  return {
    play() {}, pause() {},
    seek(x) { t = x; }, // no player to animate; setDuration reveals the full sweep
    playing() { return false; },
    setDuration(d) { t = d; onFrame(d); },
  };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Shared parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cThLif = ui.slider(sec, { key: 'threshold_lif', label: 'threshold (LIF)', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cThExp = ui.slider(sec, { key: 'threshold_exp', label: 'threshold (ELIF/AELIF)', unit: 'mV', min: -20, max: 20, step: 0.5, value: 0.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'mV', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'mV', min: -70, max: -30, step: 0.5, value: P.theta_rh });
  const cAlpha = ui.slider(sec, { key: 'alpha', label: 'alpha', unit: 'nS', min: -2, max: 2, step: 0.05, value: P.alpha });
  const cBeta = ui.slider(sec, { key: 'beta', label: 'beta', unit: 'pA', min: 0, max: 100, step: 1, value: P.beta });
  const cWTau = ui.slider(sec, { key: 'w_tau', label: 'w_tau', unit: 'ms', min: 5, max: 300, step: 5, value: P.w_tau });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });

  const fig = ui.plot(figures, { height: 300, xLabel: 'I', yLabel: 'f', xUnit: 'pA', yUnit: 'Hz' });
  // Small voltage trace of the neuron at the current sweep current I.
  const figV = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const player = makePlayer(ui, figures, frame);

  ui.note(notes, `The f-I curve plots the steady firing rate <var>f</var> (Hz) against the input current <var>I</var> (pA).
  Each point is measured by simulating that current for a fixed number of steps and counting spikes, so the
  curves are the real CoNeX dynamics, not a formula. The dots are the analytic LIF rate
  <var>f</var> = 1 / (<var>tau</var>·ln((<var>v</var><sub>ss</sub>−<var>v</var><sub>reset</sub>)/(<var>v</var><sub>ss</sub>−<var>threshold</var>))), <var>v</var><sub>ss</sub> = <var>v</var><sub>rest</sub> + <var>R</var>·<var>I</var>.
  Units: v in mV, t in ms, I in pA, R in GOhm.`);
  ui.note(notes, `For the default parameters the measured rheobases (smallest current that fires) are
  LIF ≈ 40 pA, ELIF ≈ 36 pA and AELIF ≈ 36 pA. So ELIF and AELIF start firing at a lower current than
  LIF: the exponential term pushes them over threshold sooner. AELIF's adaptation current <var>w</var> does
  not lower the onset but slows the rate once firing starts, so its curve sits below ELIF's. CoNeX has
  no refractory period, which is why the LIF reaches ~400 Hz at 120 pA. The <b>dt slider</b> shows the
  Euler discretization: a large <var>dt</var> changes the measured rate and can even change whether a neuron
  fires at all, because the update is a first-order approximation.
  ML analogy: the f-I curve is the neuron's "activation function" relating input to output rate; the
  analogy breaks because it is a dynamical, history-dependent rate, not a static function of the input.`);
  ui.code(notes, () => codeSnippet());

  function base() {
    return {
      R: cR.get(), tau: cTau.get(), v_reset: cReset.get(), v_rest: cRest.get(),
      delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: cDt.get(),
    };
  }

  // Firing rate in Hz = spikes / (total time in seconds).
  function rate(model, p, I, dt) {
    const r = simulate(model, p, I, STEPS, dt, p.v_rest);
    return r.spikes.reduce((a, b) => a + (b ? 1 : 0), 0) / (STEPS * dt / 1000);
  }

  let runData = null;

  // Recompute the f-I curves and a voltage trace per current once; the player
  // then sweeps I and reveals the curves point by point.
  function run() {
    const b = base();
    const dt = b.dt;
    const lif = { ...b, threshold: cThLif.get() };
    const elif = { ...b, threshold: cThExp.get() };
    const aelif = { ...b, threshold: cThExp.get() };
    const series = [];
    const colors = { lif: 'c1', elif: 'c2', aelif: 'c4' };
    for (const [model, p] of [['lif', lif], ['elif', elif], ['aelif', aelif]]) {
      series.push({
        x: CURRENTS.slice(),
        y: CURRENTS.map((c) => rate(model, p, c, dt)),
        label: model.toUpperCase(),
        color: colors[model],
        style: 'line',
      });
    }
    // Analytic LIF rate. lifRateAnalytic returns 1/ms (kHz); convert to Hz.
    series.push({
      x: CURRENTS.slice(),
      y: CURRENTS.map((c) => lifRateAnalytic(lif, c) * 1000),
      label: 'LIF analytic',
      color: 'ink',
      style: 'dots',
    });
    // Rate curves start at 0; pin the axis to 0 so the onset is readable.
    const maxRate = Math.max(...series.flatMap((s) => s.y), 1);
    // Precompute a real voltage trace at each current (LIF, the primary model).
    const traces = CURRENTS.map((c) => {
      const r = simulate('lif', lif, c, TRACE_STEPS, dt, lif.v_rest);
      return { c, t: r.v.map((_, n) => n * dt), v: r.v };
    });
    runData = { lif, series, maxRate, traces };
    player.setDuration(CURRENTS[CURRENTS.length - 1]);
    player.seek(0); // reveals the first frame (also covers reduced-motion, which starts paused)
  }

  // Reveal the f-I curves up to the sweep current I, and show the LIF voltage
  // trace at the largest precomputed current not above I.
  function frame(I) {
    if (!runData) return;
    const { lif, series, maxRate, traces } = runData;
    fig.draw(series, { revealX: I, yRange: [0, maxRate * 1.1] });
    let tr = traces[0];
    for (const x of traces) if (x.c <= I) tr = x;
    figV.draw([{ x: tr.t, y: tr.v, label: `v at I=${tr.c} pA`, color: 'c1', style: 'line' }],
      { hlines: [{ y: lif.threshold, label: 'threshold' }] });
  }

  function codeSnippet() {
    const b = base();
    return `# f-I: for each current I, run the neuron and count spikes over ${STEPS} steps.
from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF, AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

def fi(behavior, currents, steps=${STEPS}, dt=${b.dt}):
    rates = []
    for I in currents:
        net = Network(behavior=[TimeResolution(dt=dt)])
        pop = NeuronGroup(net=net, size=1, behavior=[behavior, Fire()])
        net.initialize(); pop.I = pop.vector(mode=I)
        net.simulate_iterations(steps)
        rates.append(pop.spikes.sum().item() / (steps * dt / 1000))   # Hz
    return rates

currents = ${JSON.stringify(CURRENTS)}   # pA
lif   = fi(LIF(R=${b.R}, threshold=${cThLif.get()}, tau=${b.tau}, v_reset=${b.v_reset}, v_rest=${b.v_rest}), currents)
elif_ = fi(ELIF(R=${b.R}, threshold=${cThExp.get()}, tau=${b.tau}, v_reset=${b.v_reset}, v_rest=${b.v_rest},
                 delta=${b.delta}, theta_rh=${b.theta_rh}), currents)
aelif = fi(AELIF(R=${b.R}, threshold=${cThExp.get()}, tau=${b.tau}, v_reset=${b.v_reset}, v_rest=${b.v_rest},
                 delta=${b.delta}, theta_rh=${b.theta_rh}, alpha=${b.alpha}, beta=${b.beta}, w_tau=${b.w_tau}), currents)`;
  }

  for (const c of [cR, cTau, cThLif, cThExp, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
