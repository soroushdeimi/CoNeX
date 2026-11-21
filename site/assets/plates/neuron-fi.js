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
const MAX_I = CURRENTS[CURRENTS.length - 1];
const FINE_STEP = 2; // pA, grid for the hero dial/bucket rate

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

  // Compute the f-I curves and a fine LIF rate curve (for the hero dial/bucket)
  // once; the player then sweeps I and reveals the curves point by point.
  let runData = null;
  function computeRun() {
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
    const maxRate = Math.max(...series.flatMap((s) => s.y), 1);
    // Fine LIF rate curve on a 2 pA grid, for the hero dial and bucket.
    const fineI = [];
    const fineRate = [];
    for (let c = 0; c <= MAX_I; c += FINE_STEP) {
      fineI.push(c);
      fineRate.push(rate('lif', lif, c, dt));
    }
    runData = { lif, series, maxRate, fineI, fineRate };
  }

  // Interpolate the fine LIF rate at an arbitrary current I (pA).
  function rateAt(I) {
    const { fineI, fineRate } = runData;
    if (I <= fineI[0]) return fineRate[0];
    if (I >= fineI[fineI.length - 1]) return fineRate[fineRate.length - 1];
    const k = I / FINE_STEP;
    const lo = Math.floor(k);
    const f = k - lo;
    return fineRate[lo] * (1 - f) + fineRate[lo + 1] * f;
  }

  // --- figures ---
  const hero = ui.hero(figures, { height: 300 });
  const fig = ui.plot(figures, { height: 300, xLabel: 'I', yLabel: 'f', xUnit: 'pA', yUnit: 'Hz' });

  // --- notes ---
  ui.note(notes, `The f-I curve plots the steady firing rate <var>f</var> (Hz) against the input current <var>I</var> (pA).
  Each point is measured by simulating that current for a fixed number of steps and counting spikes, so the
  curves are the real CoNeX dynamics, not a formula. The dots are the analytic LIF rate
  <var>f</var> = 1 / (<var>tau</var>·ln((<var>v</var><sub>ss</sub>−<var>v</var><sub>reset</sub>)/(<var>v</var><sub>ss</sub>−<var>threshold</var>))), <var>v</var><sub>ss</sub> = <var>v</var><sub>rest</sub> + <var>R</var>·<var>I</var>.
  Units: v in mV, t in ms, I in pA, R in GOhm. The hero sweeps the current with a dial; the small bucket
  fires at the measured rate <var>f</var>(<var>I</var>), and each point lands on the f-I curve as the sweep passes.`);
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

  // --- hero: current dial + small bucket firing at rate f(I) ---
  let firePhase = 0; // fractional spike accumulator for the bucket
  let lastReal = 0;
  let flash = 0; // 0..1, decays after a spike

  function drawHero(I) {
    const ctx = hero.ctx;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), coral: ui.color('c2'), ochre: ui.color('c3'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const f = rateAt(I); // Hz at the current sweep current

    // ---- left: current dial (semicircle) ----
    const dcx = W * 0.26, dcy = H * 0.62, dr = Math.min(70, W * 0.16);
    const frac = Math.max(0, Math.min(1, I / MAX_I));
    const ang = Math.PI * (1 - frac); // PI (I=0) to 0 (I=MAX)
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.arc(dcx, dcy, dr, Math.PI, 2 * Math.PI); ctx.stroke();
    // filled arc up to the current I
    ctx.strokeStyle = pal.ochre; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.arc(dcx, dcy, dr, Math.PI, ang); ctx.stroke();
    // needle
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dcx, dcy);
    ctx.lineTo(dcx + (dr - 6) * Math.cos(ang), dcy - (dr - 6) * Math.sin(ang));
    ctx.stroke();
    ctx.fillStyle = pal.ink; ctx.font = mono;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`I = ${I.toFixed(0)} pA`, dcx, dcy + 8);
    ctx.fillStyle = pal.muted; ctx.textBaseline = 'bottom';
    ctx.fillText('sweep current', dcx, dcy - dr - 8);

    // ---- right: small bucket firing at rate f(I) ----
    const bw = Math.min(120, Math.max(72, W * 0.22));
    const bx = W * 0.62, cx = bx + bw / 2;
    const yRim = H * 0.30, yBot = H * 0.82;
    // fill level oscillates: rises, then a spike drops it (schematic of firing at rate f)
    // level = (firePhase) mapped to the bucket, so it visibly fills and resets per spike
    const level = firePhase;
    const yWater = yBot - level * (yBot - yRim);
    ctx.fillStyle = flash > 0 ? pal.coral : pal.water;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(bx + 2, yWater, bw - 4, yBot - yWater);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, yRim); ctx.lineTo(bx, yBot); ctx.lineTo(bx + bw, yBot); ctx.lineTo(bx + bw, yRim);
    ctx.stroke();
    // rim flash on a spike
    if (flash > 0) {
      ctx.strokeStyle = pal.coral; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(bx, yRim); ctx.lineTo(bx + bw, yRim); ctx.stroke();
    }
    ctx.fillStyle = pal.ink; ctx.font = mono;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`f(I) = ${f.toFixed(0)} Hz`, cx, yRim - 8);
    ctx.fillStyle = pal.muted; ctx.textBaseline = 'top';
    ctx.fillText('bucket fires at f(I)', cx, yBot + 8);
  }

  // Advance the bucket's firing at the real measured rate f(I) (scaled to a few
  // flashes per second at the maximum rate so it is visible).
  function tickBucket(I, dtReal) {
    const f = rateAt(I);
    const maxF = runData.maxRate || 1;
    const realRate = (f / maxF) * 3; // spikes per real second, max 3/s
    firePhase += realRate * dtReal;
    if (firePhase >= 1) { firePhase -= 1; flash = 1; }
    flash = Math.max(0, flash - dtReal * 4);
  }

  // --- frame: the player sweeps I (0..MAX_I) ---
  let curI = 0;
  function frame(I) {
    curI = I;
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const dtReal = lastReal ? Math.min(0.05, (now - lastReal) / 1000) : 0;
    lastReal = now;
    tickBucket(I, dtReal);
    drawHero(I);
    const { series, maxRate } = runData;
    fig.draw(series, { revealX: I, yRange: [0, maxRate * 1.1], vlines: [{ x: I, label: '' }] });
    hero.narration(`sweep current I = ${I.toFixed(0)} pA → f = ${rateAt(I).toFixed(0)} Hz`);
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

  // On a parameter change: recompute the curves and re-render at the current I.
  function run() {
    computeRun();
    player.setDuration(MAX_I); // re-fires onFrame at the clamped I
  }

  computeRun();
  const player = ui.player(figures, { duration: MAX_I, unit: 'pA', playSeconds: 5, loop: true, onFrame: frame });
  frame(0);
  hero.onResize(() => frame(curI));

  for (const c of [cR, cTau, cThLif, cThExp, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt]) c.onInput(run);

  return { destroy() {} };
}
