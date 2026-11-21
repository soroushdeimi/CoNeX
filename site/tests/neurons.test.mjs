import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  AU_DEFAULTS,
  TABLE61,
  lifStep,
  elifStep,
  aelifStep,
  simulate,
  lifRheobase,
  lifRateAnalytic,
  bucketFlows,
  bucketLevel,
  splashActive,
  lifNarration,
  dropletCount,
  elifExpTerm,
  elifPhaseLine,
  elifFixedPoints,
  elifNarration,
  aelifNarration,
  fiCurve,
  kwta,
  inherentNoise,
  makeActivityHomeostasis,
  makeVoltageHomeostasis,
  homeostasisNarration,
  kwtaNarration,
  spikeTimes,
  firingStats,
} from '../assets/js/models/neurons.js';

const golden = JSON.parse(readFileSync(new URL('golden/neurons.json', import.meta.url), 'utf8'));
const S = golden.scenarios;
const TOL = 1e-4;

function close(a, b, tol = TOL, msg = '') {
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} to be within ${tol} of ${b} ${msg}`);
}

function traceClose(got, want, tol = TOL, msg = '') {
  assert.equal(got.length, want.length, `trace length mismatch ${msg}`);
  for (let i = 0; i < want.length; i++) close(got[i], want[i], tol, msg);
}

// --- LIF ---
test('lifStep matches golden lif_step trace', () => {
  const p = S.lif_step.params;
  const r = simulate('lif', p, S.lif_step.current, S.lif_step.steps, S.lif_step.dt);
  traceClose(r.v, S.lif_step.v);
  assert.deepEqual(r.spikes, S.lif_step.spikes);
});

test('lifRheobase equals (threshold - v_rest)/R and matches golden', () => {
  const p = S.lif_rheobase.params;
  close(lifRheobase(p), S.lif_rheobase.i_rheo);
});

test('bucketFlows: inflow = R*I/tau, outflow = (v - v_rest)/tau (hand-computed)', () => {
  const p = { R: 0.5, I: 65, tau: 20, v_rest: -70 };
  const { input, leak } = bucketFlows(p, -60);
  // input = 0.5 * 65 / 20 = 1.625 mV/ms; leak = (-60 - (-70)) / 20 = 0.5 mV/ms
  close(input, 1.625, 1e-9, 'input');
  close(leak, 0.5, 1e-9, 'leak');
});

test('bucketFlows: leak is 0 at rest and negative above rest; input scales with I', () => {
  const p = { R: 0.5, I: 65, tau: 20, v_rest: -70 };
  close(bucketFlows(p, -70).leak, 0, 1e-9, 'leak at rest');
  close(bucketFlows(p, -80).leak, -0.5, 1e-9, 'leak below rest');
  close(bucketFlows({ ...p, I: 0 }, -60).input, 0, 1e-9, 'no input at I=0');
});

test('bucketFlows throws on non-positive tau', () => {
  assert.throws(() => bucketFlows({ R: 0.5, I: 65, tau: 0, v_rest: -70 }, -60), /tau must be a positive finite number/);
});

test('bucketLevel: fraction between v_rest and threshold, clamped (hand-computed)', () => {
  const p = { v_rest: -70, threshold: -50 };
  // (-53.1 - (-70)) / (-50 - (-70)) = 16.9 / 20 = 0.845
  close(bucketLevel(-53.1, p), 0.845, 1e-9, 'mid level');
  close(bucketLevel(-70, p), 0, 1e-9, 'empty at v_rest');
  close(bucketLevel(-45, p), 1, 1e-9, 'clamped to rim above threshold');
  close(bucketLevel(-80, p), 0, 1e-9, 'clamped to floor below v_rest');
});

test('bucketLevel throws when threshold does not exceed v_rest', () => {
  assert.throws(() => bucketLevel(-60, { v_rest: -50, threshold: -50 }), /threshold must exceed v_rest/);
});

test('splashActive: 0.5 ms model-time window after a spike', () => {
  assert.equal(splashActive(0.4, 0, 0.5), true, '0.4 ms after -> active');
  assert.equal(splashActive(0.6, 0, 0.5), false, '0.6 ms after -> inactive');
  assert.equal(splashActive(0.5, 0, 0.5), false, 'at the window edge -> inactive');
  assert.equal(splashActive(10, null, 0.5), false, 'no spike -> inactive');
});

test('lifNarration: three regimes driven by the real flows (hand-computed)', () => {
  const p = { R: 0.5, tau: 20, v_rest: -70, threshold: -50 };
  // just spiked -> reset line
  assert.equal(lifNarration({ v: -55, p, I: 65, splashing: true }), 'threshold crossed: spike, reset to v_reset');
  // v=-60: input = 0.5*65/20 = 1.625, leak = (-60+70)/20 = 0.5 -> input > leak -> rises
  assert.equal(lifNarration({ v: -60, p, I: 65, splashing: false }), 'input beats leak: v rises');
  // v=-52: input = 1.625, leak = (-52+70)/20 = 0.9 -> still rising
  assert.equal(lifNarration({ v: -52, p, I: 65, splashing: false }), 'input beats leak: v rises');
  // v=-40 (above steady state -37.5? no, vss=-37.5): leak = (-40+70)/20 = 1.5 < 1.625 -> rising
  // use a low I so leak exceeds input: I=10 -> input=0.25; v=-50 -> leak=1.0 -> settles
  assert.equal(lifNarration({ v: -50, p, I: 10, splashing: false }), 'leak balances input: v settles below threshold');
});

test('dropletCount: proportional to flow, zero at no flow, capped, seeded', () => {
  const rng = { next: () => 0.99 }; // always rounds up the fractional part
  assert.equal(dropletCount(0, 0.016, rng), 0, 'no flow -> no droplets');
  assert.equal(dropletCount(-1, 0.016, rng), 0, 'negative flow -> no droplets');
  // flow=10 mV/ms, frame 0.016 s, rate 30 -> expected 4.8 -> 4 + (0.99<0.8? no) = 4
  assert.equal(dropletCount(10, 0.016, rng), 4, 'whole part, fraction not met');
  // fraction met: expected 4.2 with rng 0.1 -> 4 + 1 = 5
  assert.equal(dropletCount(10, 0.014, { next: () => 0.1 }), 5, 'fractional part met');
  // capped at 40
  assert.equal(dropletCount(1000, 1, rng), 40, 'capped');
});

test('LIF: N live steps equal the precomputed run bit-for-bit', () => {
  // The plate's live step() calls lifStep once per dt with the current params.
  // Stepping it N times must reproduce simulate() exactly (same order, same math).
  const p = { R: 0.5, tau: 20, threshold: -50, v_reset: -55, v_rest: -70, dt: 0.1 };
  const I = 65;
  const N = 500;
  const ref = simulate('lif', p, I, N, p.dt, p.v_rest);
  let state = { v: p.v_rest };
  const liveV = [], liveSpikes = [];
  for (let i = 0; i < N; i++) {
    const r = lifStep(state, p, I, p.dt);
    state = { v: r.v };
    liveV.push(r.v);
    liveSpikes.push(r.spike);
  }
  assert.deepEqual(liveV, ref.v, 'live v trace must match the precomputed run exactly');
  assert.deepEqual(liveSpikes, ref.spikes, 'live spike flags must match exactly');
});

test('LIF at rheobase never fires; just above it fires (golden)', () => {
  const p = S.lif_rheobase.params;
  const at = simulate('lif', p, S.lif_rheobase.at_rheobase.current, S.lif_rheobase.at_rheobase.steps);
  assert.equal(at.spikes.filter(Boolean).length, S.lif_rheobase.at_rheobase.spike_count);
  close(at.v.at(-1), S.lif_rheobase.at_rheobase.v_final);
  const above = simulate('lif', p, S.lif_rheobase.above_rheobase.current, S.lif_rheobase.above_rheobase.steps);
  assert.equal(above.spikes.filter(Boolean).length, S.lif_rheobase.above_rheobase.spike_count);
  assert.equal(above.spikes.indexOf(true), S.lif_rheobase.above_rheobase.first_spike_step);
});

test('lifRateAnalytic is 0 at/below rheobase and positive above', () => {
  const p = S.lif_rheobase.params;
  const ir = lifRheobase(p);
  assert.equal(lifRateAnalytic(p, ir), 0.0);
  assert.equal(lifRateAnalytic(p, ir - 0.1), 0.0);
  assert.ok(lifRateAnalytic(p, ir + 0.1) > 0);
});

// --- ELIF ---
test('elifStep matches golden elif_step trace', () => {
  const p = S.elif_step.params;
  const r = simulate('elif', p, S.elif_step.current, S.elif_step.steps, S.elif_step.dt);
  traceClose(r.v, S.elif_step.v);
  assert.deepEqual(r.spikes, S.elif_step.spikes);
});

test('elifExpTerm: delta*exp((v-theta_rh)/delta) (hand-computed)', () => {
  const p = { delta: 2, theta_rh: -50 };
  // v = theta_rh -> delta*exp(0) = delta = 2
  close(elifExpTerm(-50, p), 2.0, 1e-9, 'at theta_rh');
  // v = theta_rh + delta -> delta*exp(1) = 2*e
  close(elifExpTerm(-48, p), 2 * Math.E, 1e-9, 'one delta above');
  // v = theta_rh - 10*delta -> delta*exp(-10) = 2*4.54e-5 = 9.08e-5 (tiny)
  close(elifExpTerm(-70, p), 2 * Math.exp(-10), 1e-9, 'far below theta_rh');
});

test('elifExpTerm throws on non-positive delta', () => {
  assert.throws(() => elifExpTerm(0, { delta: 0, theta_rh: -50 }), /delta must be a positive finite number/);
});

test('elifPhaseLine: f(v) = (-(v-v_rest) + delta*exp((v-theta_rh)/delta) + R*I)/tau (hand-computed)', () => {
  const p = { R: 0.5, tau: 20, v_rest: -70, delta: 2, theta_rh: -50 };
  // v = v_rest: leak term 0, exp term = 2*exp(-10) = 9.08e-5, R*I = 0.5*20 = 10
  // f = (0 + 9.08e-5 + 10)/20 = 0.5000045 mV/ms
  close(elifPhaseLine(-70, p, 20), (0 + 2 * Math.exp(-10) + 10) / 20, 1e-9, 'at v_rest');
  // v = theta_rh: exp term = delta = 2; leak = -(-50+70) = -20; R*I = 10
  // f = (-20 + 2 + 10)/20 = -0.4 mV/ms
  close(elifPhaseLine(-50, p, 20), (-20 + 2 + 10) / 20, 1e-9, 'at theta_rh');
});

test('elifFixedPoints: fixed points by hand for the defaults (stable/unstable)', () => {
  const p = { R: 0.5, tau: 20, v_rest: -70, delta: 2, theta_rh: -50 };
  const vMin = p.v_rest - 5, vMax = 0;
  // I = 20: two fixed points, the lower one stable, the upper one unstable.
  const fp20 = elifFixedPoints(p, 20, vMin, vMax);
  assert.equal(fp20.length, 2, 'I=20 has two fixed points');
  close(fp20[0].v, -59.9864, 1e-3, 'lower fixed point');
  assert.equal(fp20[0].stable, true, 'lower fixed point is stable');
  close(fp20[1].v, -46.1263, 1e-3, 'upper fixed point');
  assert.equal(fp20[1].stable, false, 'upper fixed point is unstable');
  // I = 65: no fixed point below threshold (the neuron fires).
  assert.equal(elifFixedPoints(p, 65, vMin, vMax).length, 0, 'I=65 has no fixed point');
});

test('elifNarration: take-off vs balance, driven by the real terms (hand-computed)', () => {
  const p = { R: 0.5, tau: 20, v_rest: -70, threshold: 0, delta: 2, theta_rh: -50 };
  const I = 65;
  // just spiked -> reset line
  assert.equal(elifNarration({ v: -55, p, I, splashing: true }), 'threshold crossed: spike, reset to v_reset');
  // v=-50: exp term = 2 mV -> /tau = 0.1 mV/ms; input = 0.5*65/20 = 1.625 -> balance
  assert.equal(elifNarration({ v: -50, p, I, splashing: false }), 'v below take-off: input and leak balance');
  // v=-40: exp term = 2*exp(5) = 204 mV -> /tau = 10.2 mV/ms > input 1.625 -> take-off
  assert.equal(elifNarration({ v: -40, p, I, splashing: false }), 'exponential term takes over: v accelerates to threshold');
});

test('ELIF: N live steps equal the precomputed run bit-for-bit', () => {
  const p = { R: 0.5, tau: 20, threshold: 0, v_reset: -55, v_rest: -70, delta: 2, theta_rh: -50, dt: 0.1 };
  const I = 65;
  const N = 500;
  const ref = simulate('elif', p, I, N, p.dt, p.v_rest);
  let state = { v: p.v_rest };
  const liveV = [], liveSpikes = [];
  for (let i = 0; i < N; i++) {
    const r = elifStep(state, p, I, p.dt);
    state = { v: r.v };
    liveV.push(r.v);
    liveSpikes.push(r.spike);
  }
  assert.deepEqual(liveV, ref.v, 'live v trace must match the precomputed run exactly');
  assert.deepEqual(liveSpikes, ref.spikes, 'live spike flags must match exactly');
});

// --- AELIF ---
test('aelifStep matches golden aelif_step trace (v and omega)', () => {
  const p = S.aelif_step.params;
  const r = simulate('aelif', p, S.aelif_step.current, S.aelif_step.steps, S.aelif_step.dt);
  traceClose(r.v, S.aelif_step.v);
  traceClose(r.omega, S.aelif_step.omega);
  assert.deepEqual(r.spikes, S.aelif_step.spikes);
});

test('aelifNarration: adaptation vs input, driven by the real quantities (hand-computed)', () => {
  const p = { R: 0.5, tau: 20, v_rest: -70, threshold: 0 };
  const I = 65;
  // just spiked -> reset line
  assert.equal(aelifNarration({ v: -55, omega: 5, p, I, splashing: true }), 'threshold crossed: spike, reset to v_reset');
  // w below I -> input wins -> rises
  assert.equal(aelifNarration({ v: -60, omega: 10, p, I, splashing: false }), 'input beats adaptation: v rises toward threshold');
  // w above I -> adaptation wins -> pulled down
  assert.equal(aelifNarration({ v: -60, omega: 80, p, I, splashing: false }), 'adaptation w exceeds input: v is pulled down, firing slows');
});

test('AELIF: N live steps equal the precomputed run bit-for-bit', () => {
  const p = { R: 0.5, tau: 20, threshold: 0, v_reset: -55, v_rest: -70, delta: 2, theta_rh: -50, alpha: 0, beta: 60, w_tau: 30, dt: 0.1 };
  const I = 65;
  const N = 500;
  const ref = simulate('aelif', p, I, N, p.dt, p.v_rest);
  let state = { v: p.v_rest, omega: 0.0 };
  const liveV = [], liveW = [], liveSpikes = [];
  for (let i = 0; i < N; i++) {
    const r = aelifStep(state, p, I, p.dt);
    state = { v: r.v, omega: r.omega };
    liveV.push(r.v);
    liveW.push(r.omega);
    liveSpikes.push(r.spike);
  }
  assert.deepEqual(liveV, ref.v, 'live v trace must match the precomputed run exactly');
  assert.deepEqual(liveW, ref.omega, 'live w trace must match the precomputed run exactly');
  assert.deepEqual(liveSpikes, ref.spikes, 'live spike flags must match exactly');
});

// --- f-I curves ---
test('fiCurve matches golden counts for lif, elif, aelif', () => {
  const { currents, steps, dt } = S.fi_curves;
  for (const model of ['lif', 'elif', 'aelif']) {
    const p = model === 'lif' ? S.lif_step.params
      : model === 'elif' ? S.elif_step.params
      : S.aelif_step.params;
    const got = fiCurve(model, p, currents, steps, dt);
    assert.deepEqual(got, S.fi_curves.counts[model], `fiCurve mismatch for ${model}`);
  }
});

// --- KWTA ---
test('kwta matches golden cases (top-k keep v, losers reset)', () => {
  const { threshold, v_reset } = S.kwta.params;
  const k = S.kwta.kwta.k;
  for (const c of S.kwta.cases) {
    assert.deepEqual(kwta(c.v_in, threshold, v_reset, k), c.v_out, `kwta case ${JSON.stringify(c.v_in)}`);
  }
});

test('kwta throws on non-integer k', () => {
  assert.throws(() => kwta([1, 2], 0, 0, 1.5), /k must be a non-negative integer/);
});

test('kwtaNarration: winners vs inhibition, driven by the real count (hand-computed)', () => {
  assert.equal(kwtaNarration({ nAbove: 5, k: 2 }), '5 above threshold: top 2 win, the rest are reset');
  assert.equal(kwtaNarration({ nAbove: 2, k: 2 }), '2 above threshold (≤ 2): all fire, no inhibition');
  assert.equal(kwtaNarration({ nAbove: 0, k: 2 }), 'no neuron above threshold');
});

// --- InherentNoise ---
test('inherentNoise adds uniform[0,1)*scale+offset', () => {
  // Deterministic rng: returns 0.5 each call.
  const rng = { next: () => 0.5 };
  assert.deepEqual(inherentNoise([0, 10], rng, 2.0, 1.0), [2.0, 12.0]);
});

// --- Activity homeostasis (group of 4 identical LIF, current 3.0) ---
test('activity homeostasis matches golden threshold_trace and window_spikes', () => {
  const p = S.homeostasis_activity.params;
  const h = S.homeostasis_activity.homeostasis;
  const size = S.homeostasis_activity.size;
  const I = S.homeostasis_activity.current;
  const dt = S.homeostasis_activity.dt;
  const steps = S.homeostasis_activity.steps;

  const hos = Array.from({ length: size }, () => makeActivityHomeostasis(h));
  const vs = new Array(size).fill(0.0);
  const thresholds = new Array(size).fill(p.threshold);
  const thresholdTrace = [];
  const windowSpikes = [];
  let acc = 0;
  for (let it = 1; it <= steps; it++) {
    let groupSpikes = 0;
    for (let j = 0; j < size; j++) {
      const r = lifStep({ v: vs[j] }, p, I, dt);
      vs[j] = r.v;
      groupSpikes += r.spike ? 1 : 0;
      thresholds[j] = hos[j].step(r.spike, thresholds[j], it);
    }
    acc += groupSpikes;
    if (it % h.window_size === 0) {
      thresholdTrace.push(thresholds[0]);
      windowSpikes.push(acc);
      acc = 0;
    }
  }
  assert.equal(thresholdTrace.length, S.homeostasis_activity.threshold_trace.length);
  for (let w = 0; w < thresholdTrace.length; w++) {
    close(thresholdTrace[w], S.homeostasis_activity.threshold_trace[w][0]);
  }
  assert.deepEqual(windowSpikes, S.homeostasis_activity.window_spikes);
});

test('activity homeostasis throws when window_size <= activity_rate', () => {
  assert.throws(
    () => makeActivityHomeostasis({ activity_rate: 5, window_size: 5, updating_rate: 0.1 }),
    /window_size must exceed activity_rate/
  );
});

test('homeostasisNarration: rate vs target, driven by the real quantities (hand-computed)', () => {
  assert.equal(homeostasisNarration({ rate: 20, target: 10, mode: 'activity' }), 'rate above target: threshold rises to slow firing');
  assert.equal(homeostasisNarration({ rate: 5, target: 10, mode: 'activity' }), 'rate below target: threshold falls to speed firing');
  assert.equal(homeostasisNarration({ rate: 10, target: 10, mode: 'activity' }), 'rate at target: threshold holds');
  assert.equal(homeostasisNarration({ rate: 20, target: 10, mode: 'voltage' }), 'voltage pushed toward the target band');
});

test('homeostasis: N live steps equal the precomputed run bit-for-bit', () => {
  // The plate's live step() runs the same per-neuron LIF + homeostasis as the
  // golden test. Stepping it N times must reproduce the golden window_spikes
  // exactly (the precomputed run), for a fixed parameter set.
  const p = S.homeostasis_activity.params;
  const h = S.homeostasis_activity.homeostasis;
  const size = S.homeostasis_activity.size;
  const I = S.homeostasis_activity.current;
  const dt = S.homeostasis_activity.dt;
  const N = S.homeostasis_activity.steps;
  const hos = Array.from({ length: size }, () => makeActivityHomeostasis(h));
  const vs = new Array(size).fill(0.0);
  const thresholds = new Array(size).fill(p.threshold);
  const windowSpikes = [];
  let acc = 0;
  for (let it = 1; it <= N; it++) {
    let groupSpikes = 0;
    for (let j = 0; j < size; j++) {
      const r = lifStep({ v: vs[j] }, { ...p, threshold: thresholds[j] }, I, dt);
      vs[j] = r.v;
      groupSpikes += r.spike ? 1 : 0;
      thresholds[j] = hos[j].step(r.spike, thresholds[j], it);
    }
    acc += groupSpikes;
    if (it % h.window_size === 0) { windowSpikes.push(acc); acc = 0; }
  }
  assert.deepEqual(windowSpikes, S.homeostasis_activity.window_spikes, 'live window spikes must equal the precomputed run');
});

// --- Voltage homeostasis (single LIF, current 1.0, band at target 5.0) ---
test('voltage homeostasis matches golden v and exhaustion traces', () => {
  const p = S.homeostasis_voltage.params;
  const h = S.homeostasis_voltage.homeostasis;
  const I = S.homeostasis_voltage.current;
  const dt = S.homeostasis_voltage.dt;
  const steps = S.homeostasis_voltage.steps;

  const ho = makeVoltageHomeostasis(h);
  let v = 0.0;
  const vTrace = [];
  const exhTrace = [];
  for (let it = 1; it <= steps; it++) {
    const r = lifStep({ v }, p, I, dt);
    v = r.v;
    v = ho.step(v);
    vTrace.push(v);
    exhTrace.push(ho.exhaustion);
  }
  traceClose(vTrace, S.homeostasis_voltage.v);
  traceClose(exhTrace, S.homeostasis_voltage.exhaustion);
});

// --- Physiological cases (Gerstner Table 6.1 units) ---
// The golden physiological networks are built with dtype=torch.float64, so the JS
// (float64) mirror matches to ~1e-12 mV. Tolerance 1e-6 mV is far below any
// parameter error (a 1% tau or R error shifts the trace by mV), so a wrong
// parameter fails. Spike steps are compared exactly.
const PHYS_TOL = 1e-6; // mV, max |dv| over a 5000-step physiological trace

test('physiological LIF matches golden trace (Table 6.1 units)', () => {
  const g = S.phys_lif;
  const r = simulate('lif', g.params, g.current, g.steps, g.dt, g.params.v_rest);
  traceClose(r.v, g.v, PHYS_TOL);
  assert.deepEqual(r.spikes, g.spikes);
});

test('physiological ELIF matches golden trace (Table 6.1 units)', () => {
  const g = S.phys_elif;
  const r = simulate('elif', g.params, g.current, g.steps, g.dt, g.params.v_rest);
  traceClose(r.v, g.v, PHYS_TOL);
  assert.deepEqual(r.spikes, g.spikes);
});

test('physiological AELIF matches each Table 6.1 row trace and spike steps', () => {
  for (const [name, g] of Object.entries(S.phys_aelif)) {
    const r = simulate('aelif', g.params, g.current, g.steps, g.dt, g.params.v_rest);
    traceClose(r.v, g.v, PHYS_TOL, `AELIF ${name} v trace`);
    assert.deepEqual(r.spikes, g.spikes, `AELIF ${name} spike steps`);
  }
});

// Per-row firing-pattern signatures. A row is "reproduced" only if its measured
// signature holds; otherwise it is labelled not-reproduced (see DOUBTS).
function rowStats(name) {
  const g = S.phys_aelif[name];
  const r = simulate('aelif', g.params, g.current, g.steps, g.dt, g.params.v_rest);
  return firingStats(spikeTimes(r.spikes, g.dt));
}

test('Tonic: ISIs nearly constant (low CV)', () => {
  const s = rowStats('Tonic');
  assert.ok(s.cv < 0.1, `Tonic CV=${s.cv} should be < 0.1`);
});

test('Adapting: ISIs increase over time (late > early)', () => {
  const s = rowStats('Adapting');
  assert.ok(s.late > s.early, `Adapting late ISI ${s.late} should exceed early ${s.early}`);
});

test('Initial burst: first ISIs much shorter than late ISIs', () => {
  const s = rowStats('Initial burst');
  assert.ok(s.early < 0.6 * s.late, `Initial burst early ${s.early} should be < 0.6*late ${s.late}`);
});

test('Transient: ISIs increase and firing stops (few spikes, late >> early)', () => {
  const s = rowStats('Transient');
  assert.ok(s.n < 40, `Transient should fire few spikes, got ${s.n}`);
  assert.ok(s.late > 1.5 * s.early, `Transient late ${s.late} should exceed 1.5*early ${s.early}`);
});

test('Delayed: first spike much later than Tonic', () => {
  const d = rowStats('Delayed');
  const t = rowStats('Tonic');
  assert.ok(d.first_t > 3 * t.first_t, `Delayed first spike ${d.first_t} should be > 3*Tonic ${t.first_t}`);
});

// Bursting and Irregular do NOT reproduce their named patterns with CoNeX's AELIF
// at dt=0.1 ms: both fire nearly regularly (low CV, no ISI clusters). These tests
// assert the ACTUAL (non-bursting, non-irregular) behavior so the plate can label
// the rows honestly. See DOUBTS.
test('Bursting: NOT reproduced (fires nearly regularly, low CV)', () => {
  const s = rowStats('Bursting');
  assert.ok(s.cv < 0.2, `Bursting CV=${s.cv} is low -> not a two-cluster ISI distribution`);
});

test('Irregular: NOT reproduced (fires nearly regularly, low CV)', () => {
  const s = rowStats('Irregular');
  assert.ok(s.cv < 0.2, `Irregular CV=${s.cv} is low -> ISIs are not irregular`);
});

// CoNeX's Euler step multiplies the whole adaptation update by dt/w_tau, so each
// spike raises omega by beta*dt, not by beta as in Gerstner's AdEx. With the
// jump restored to beta (beta scaled by 1/dt), the named patterns come back:
// the ISI CV of Bursting and Irregular rises above 1, while the CoNeX jump
// keeps them below 0.2.
function rowCv(name, betaScale) {
  const g = S.phys_aelif[name];
  const r = simulate('aelif', { ...g.params, beta: g.params.beta * betaScale }, g.current, g.steps, g.dt, g.params.v_rest);
  return firingStats(spikeTimes(r.spikes, g.dt)).cv;
}

test('Bursting: ISI CV > 1 with jump b (beta/dt), < 0.2 with the CoNeX jump', () => {
  assert.ok(rowCv('Bursting', 1 / 0.1) > 1, `Bursting CV=${rowCv('Bursting', 1 / 0.1)} should be > 1 with jump b`);
  assert.ok(rowCv('Bursting', 1) < 0.2, `Bursting CV=${rowCv('Bursting', 1)} should be < 0.2 with the CoNeX jump`);
});

test('Irregular: ISI CV > 1 with jump b (beta/dt), < 0.2 with the CoNeX jump', () => {
  assert.ok(rowCv('Irregular', 1 / 0.1) > 1, `Irregular CV=${rowCv('Irregular', 1 / 0.1)} should be > 1 with jump b`);
  assert.ok(rowCv('Irregular', 1) < 0.2, `Irregular CV=${rowCv('Irregular', 1)} should be < 0.2 with the CoNeX jump`);
});
