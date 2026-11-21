import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  HH_DEFAULTS,
  hhRates,
  hhSteadyGates,
  hhStep,
  hhSimulate,
  flapAngle,
  particleRate,
  membranePolarity,
  hhNarration,
} from '../assets/js/models/hh.js';

const golden = JSON.parse(readFileSync(new URL('golden/hh.json', import.meta.url), 'utf8'));
const S = golden.scenarios;
const P = golden.params;
const DT = golden.dt;

const V_TOL = 1e-6;   // mV
const GATE_TOL = 1e-9;

function close(a, b, tol, msg = '') {
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} to be within ${tol} of ${b} ${msg}`);
}
function traceClose(got, want, tol, msg = '') {
  assert.equal(got.length, want.length, `trace length mismatch ${msg}`);
  for (let i = 0; i < want.length; i++) close(got[i], want[i], tol, msg);
}

// --- Golden: JS mirror vs the real CoNeX HH ---

test('rest (I=0) matches golden v and gates', () => {
  const r = hhSimulate(P, 0.0, S.rest.steps, DT);
  traceClose(r.v, S.rest.v, V_TOL, 'rest v');
  traceClose(r.m, S.rest.m, GATE_TOL, 'rest m');
  traceClose(r.h, S.rest.h, GATE_TOL, 'rest h');
  traceClose(r.n, S.rest.n, GATE_TOL, 'rest n');
  assert.deepEqual(r.spikes, S.rest.spikes);
});

test('2 ms pulse matches golden v, gates, currents and spike steps', () => {
  const { pulse_amplitude: A, pulse_steps: ps } = S.pulse;
  const r = hhSimulate(P, (i) => (i >= 2000 && i < 2000 + ps ? A : 0.0), S.pulse.steps, DT);
  traceClose(r.v, S.pulse.v, V_TOL, 'pulse v');
  traceClose(r.m, S.pulse.m, GATE_TOL, 'pulse m');
  traceClose(r.n, S.pulse.n, GATE_TOL, 'pulse n');
  traceClose(r.i_Na, S.pulse.i_Na, V_TOL, 'pulse i_Na');
  traceClose(r.i_K, S.pulse.i_K, V_TOL, 'pulse i_K');
  assert.deepEqual(r.spikes, S.pulse.spikes);
});

test('constant 10 uA/cm2 for 50 ms matches golden (repetitive firing)', () => {
  const r = hhSimulate(P, 10.0, S.const_10.steps, DT);
  traceClose(r.v, S.const_10.v, V_TOL, 'const_10 v');
  traceClose(r.m, S.const_10.m, GATE_TOL, 'const_10 m');
  assert.deepEqual(r.spikes, S.const_10.spikes);
  assert.equal(r.spikes.filter(Boolean).length, S.const_10.spikes.filter(Boolean).length);
});

test('constant 5 uA/cm2 for 200 ms matches golden (onset only)', () => {
  const r = hhSimulate(P, 5.0, S.const_5.steps, DT);
  traceClose(r.v, S.const_5.v, V_TOL, 'const_5 v');
  assert.deepEqual(r.spikes, S.const_5.spikes);
  assert.equal(r.spikes.filter(Boolean).length, S.const_5.spikes.filter(Boolean).length);
});

// --- Gate rates and steady states (hand-computed) ---

test('hhRates at -65 mV matches hand-computed HH (1952) rates', () => {
  const r = hhRates(-65);
  // u = (-65+40)/10 = -2.5; alpha_m = -u/expm1(-u) = 2.5/expm1(2.5)
  close(r.alpha_m, 2.5 / Math.expm1(2.5), 1e-12, 'alpha_m');
  close(r.beta_m, 4 * Math.exp(0), 1e-12, 'beta_m at -65 = 4');
  close(r.alpha_h, 0.07 * Math.exp(0), 1e-12, 'alpha_h at -65 = 0.07');
  close(r.beta_h, 1 / (1 + Math.exp(3)), 1e-12, 'beta_h');
  close(r.alpha_n, 0.1 * (1 / Math.expm1(1)), 1e-12, 'alpha_n'); // q = -1
  close(r.beta_n, 0.125 * Math.exp(0), 1e-12, 'beta_n at -65 = 0.125');
});

test('hhRates is finite at the singular voltages -40 and -55 mV', () => {
  for (const v of [-40, -55]) {
    const r = hhRates(v);
    for (const k of Object.keys(r)) assert.ok(Number.isFinite(r[k]), `rate ${k} finite at ${v}`);
  }
  // alpha_m(-40) = 1 (the Taylor limit); alpha_n(-55) = 0.1
  close(hhRates(-40).alpha_m, 1.0, 1e-6, 'alpha_m limit at -40');
  close(hhRates(-55).alpha_n, 0.1, 1e-6, 'alpha_n limit at -55');
});

test('hhSteadyGates at -65 mV matches hand-computed alpha/(alpha+beta)', () => {
  const g = hhSteadyGates(-65);
  const r = hhRates(-65);
  close(g.m, r.alpha_m / (r.alpha_m + r.beta_m), 1e-12, 'm_inf');
  close(g.h, r.alpha_h / (r.alpha_h + r.beta_h), 1e-12, 'h_inf');
  close(g.n, r.alpha_n / (r.alpha_n + r.beta_n), 1e-12, 'n_inf');
  // m_inf ~ 0.0529, h_inf ~ 0.5961, n_inf ~ 0.3177 (measured)
  close(g.m, 0.0529, 1e-3, 'm_inf magnitude');
  close(g.h, 0.5961, 1e-3, 'h_inf magnitude');
  close(g.n, 0.3177, 1e-3, 'n_inf magnitude');
});

test('hhStep: no reset after a spike (v keeps rising past threshold)', () => {
  // A state well above threshold with strong inward Na current: v must keep
  // increasing (no reset to a low value), unlike LIF.
  const p = { ...P, threshold: 0 };
  const state = { v: 5, m: 0.5, h: 0.5, n: 0.3 };
  const r = hhStep(state, p, 0, DT);
  assert.ok(r.v > state.v, `v should rise without reset, got ${r.v} from ${state.v}`);
});

// --- Pure mapping functions (AMENDMENT 5) ---

test('flapAngle: gate variable -> opening angle (hand-computed)', () => {
  assert.equal(flapAngle(0), 0);
  assert.equal(flapAngle(1), 50);
  assert.equal(flapAngle(0.5), 25);
  assert.equal(flapAngle(0.2), 10);
});

test('flapAngle throws on a gate outside [0,1]', () => {
  assert.throws(() => flapAngle(1.2), /gate must be in \[0,1\]/);
  assert.throws(() => flapAngle(-0.1), /gate must be in \[0,1\]/);
});

test('particleRate: proportional to |current| (hand-computed)', () => {
  assert.equal(particleRate(0, 1), 0);
  assert.equal(particleRate(10, 1), 10);
  assert.equal(particleRate(-10, 1), 10); // sign dropped
  assert.equal(particleRate(10, 2), 20);  // scale
});

test('membranePolarity: voltage -> [-1,1] (hand-computed)', () => {
  close(membranePolarity(-65), 0, 1e-12, 'neutral at rest');
  close(membranePolarity(-90), -1, 1e-12, 'fully hyperpolarized');
  close(membranePolarity(40), 1, 1e-12, 'fully depolarized');
  close(membranePolarity(-77.5), -0.5, 1e-12, 'mid hyperpolarized'); // (-77.5+65)/25
  close(membranePolarity(10), 75 / 105, 1e-12, 'mid depolarized');   // (10+65)/105
  close(membranePolarity(-100), -1, 1e-12, 'clamped below');
  close(membranePolarity(100), 1, 1e-12, 'clamped above');
});

test('hhNarration: driven by the model state (hand-computed)', () => {
  const base = { v_rest: -65, i_Na: 0, i_K: 0, h: 0.6 };
  assert.equal(hhNarration({ ...base, v: -75 }), 'afterhyperpolarization: v below rest, K+ still flowing out');
  assert.equal(hhNarration({ ...base, v: 10, i_Na: -60, h: 0.6 }), 'Na+ rushes in: strong inward current');
  assert.equal(hhNarration({ ...base, v: -60, i_K: 30 }), 'K+ gates open: repolarization');
  assert.equal(hhNarration({ ...base, v: 10, i_Na: -30, h: 0.3 }), 'h closes: Na+ inactivates');
  assert.equal(hhNarration({ ...base, v: 10, i_Na: -30, h: 0.6 }), 'Na+ gates open: depolarization accelerates');
  assert.equal(hhNarration({ ...base, v: -65 }), 'at rest');
});
