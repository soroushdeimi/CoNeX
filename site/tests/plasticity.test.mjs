// Tests for site/assets/js/models/plasticity.js.
// Deterministic parts are compared against site/tests/golden/plasticity.json
// (produced by scripts/site/golden_plasticity.py from the real CoNeX classes).
// The Poisson encoders use torch's Philox RNG in the golden, which the site's
// mulberry32 PRNG cannot reproduce, so those are tested by property instead.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import * as M from '../assets/js/models/plasticity.js';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, 'golden', 'plasticity.json'), 'utf8'));

const close = (a, b, eps = 1e-5) => Math.abs(a - b) <= eps;
const closeArr = (a, b, eps = 1e-5) =>
  a.length === b.length && a.every((x, i) => close(x, b[i], eps));
const sameBool = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

// Replay a scripted-spike STDP/iSTDP/RSTDP run and compare against the golden.
function replayStdp(g, rule) {
  const p = g.params;
  const steps = g.pre_spikes.length;
  const nSrc = g.pre_spikes[0].length;
  const nDst = g.post_spikes_in[0].length;
  let w = new Array(nSrc * nDst).fill(p.w0);
  let preT = new Array(nSrc).fill(0);
  let postT = new Array(nDst).fill(0);
  for (let t = 0; t < steps; t++) {
    const preS = g.pre_spikes[t];
    const postS = g.post_spikes_in[t];
    preT = preT.map((v, i) => M.stepTrace(v, preS[i], { tau_s: p.tau_s, dt: p.dt }));
    postT = postT.map((v, i) => M.stepTrace(v, postS[i], { tau_s: p.tau_s, dt: p.dt }));
    const dw = rule({
      weights: w, pre_spike: preS, post_spike: postS, pre_trace: preT, post_trace: postT,
      a_plus: p.a_plus, a_minus: p.a_minus, w_min: p.w_min, w_max: p.w_max,
      positive_bound: p.positive_bound, negative_bound: p.negative_bound,
      rho: p.rho, lr: p.lr, is_inhibitory: p.is_inhibitory, tau_s: p.tau_s,
    });
    w = w.map((x, i) => x + dw[i]);
    assert.ok(closeArr(w, g.weights[t]), `weight mismatch at step ${t}`);
    assert.ok(closeArr(preT, g.pre_trace[t]), `pre_trace mismatch at step ${t}`);
    assert.ok(closeArr(postT, g.post_trace[t]), `post_trace mismatch at step ${t}`);
  }
}

test('SimpleSTDP replays the golden weight and trace trajectory', () => {
  replayStdp(golden.stdp, M.stdpDw);
});

test('SimpleiSTDP alpha equals 2*rho*tau_s/1000 and replays the golden weights', () => {
  const i = golden.istdp;
  assert.ok(close(M.istdpAlpha({ rho: i.params.rho, tau_s: i.params.tau_s }), i.alpha));
  replayStdp(i, M.istdpDw);
});

test('SimpleRSTDP with reward replays the golden eligibility, dopamine and weights', () => {
  const r = golden.rstdp_reward;
  const p = r.params;
  const nSrc = r.pre_spikes[0].length;
  const nDst = r.post_spikes_in[0].length;
  let w = new Array(nSrc * nDst).fill(p.w0);
  let c = new Array(nSrc * nDst).fill(0);
  let dop = 0;
  let preT = new Array(nSrc).fill(0);
  let postT = new Array(nDst).fill(0);
  for (let t = 0; t < r.pre_spikes.length; t++) {
    dop = M.dopamineStep({ dopamine_concentration: dop, tau_dopamine: p.tau_dopamine, payoff: r.payoff[t] });
    const preS = r.pre_spikes[t];
    const postS = r.post_spikes_in[t];
    preT = preT.map((v, i) => M.stepTrace(v, preS[i], { tau_s: p.tau_s, dt: p.dt }));
    postT = postT.map((v, i) => M.stepTrace(v, postS[i], { tau_s: p.tau_s, dt: p.dt }));
    const dw = M.stdpDw({ weights: w, pre_spike: preS, post_spike: postS, pre_trace: preT, post_trace: postT, a_plus: p.a_plus, a_minus: p.a_minus });
    const step = M.rstdpStep({ c, dw, tau_c: p.tau_c, dopamine_concentration: dop });
    c = step.c;
    w = w.map((x, i) => x + step.dwApplied[i]);
    assert.ok(close(dop, r.dopamine[t], 1e-5), `dopamine mismatch at step ${t}`);
    assert.ok(closeArr(c, r.eligibility[t]), `eligibility mismatch at step ${t}`);
    assert.ok(closeArr(w, r.weights[t]), `weight mismatch at step ${t}`);
  }
});

test('SimpleRSTDP without reward leaves weights unchanged (no dopamine)', () => {
  const r = golden.rstdp_noreward;
  // with zero payoff the dopamine stays 0, so weights must equal the init value
  for (const row of r.weights) {
    for (const x of row) assert.ok(close(x, r.params.w0, 1e-9));
  }
  // and the reward run must have moved at least one weight
  const moved = golden.rstdp_reward.weights.flat().some((x) => !close(x, r.params.w0, 1e-9));
  assert.ok(moved, 'reward run should change a weight');
});

// --- encoders ---

test('Intensity2Latency matches the golden exactly (deterministic)', () => {
  const e = golden.encoders;
  const out = M.intensity2Latency(e.image, { time_window: e.params.time_window });
  // golden stores 0/1 floats; compare by truthiness
  assert.ok(out.length === e.intensity2latency.length);
  assert.ok(out.every((x, i) => Boolean(x) === (e.intensity2latency[i] !== 0)));
});

test('Intensity2Latency: stronger pixels spike sooner (lower latency)', () => {
  // a single bright pixel spikes earlier than a dim one
  const img = [0.2, 0.8];
  const out = M.intensity2Latency(img, { time_window: 20 });
  const firstSpike = (i) => {
    for (let t = 0; t < 20; t++) if (out[t * 2 + i]) return t;
    return -1;
  };
  assert.ok(firstSpike(1) < firstSpike(0), 'brighter pixel should spike sooner');
});

test('Intensity2Latency: below-threshold pixels never spike', () => {
  const img = [0.0, 0.5];
  const out = M.intensity2Latency(img, { time_window: 20, threshold: 0.3 });
  assert.ok(out.every((x, i) => i % 2 === 0 ? !x : true), 'below-threshold pixel must not spike');
});

test('SimplePoisson: expected spike count scales with intensity and ratio', () => {
  const n = 1000;
  const img = new Array(n).fill(0.5);
  const tw = 100;
  const ratio = 0.5;
  const out = M.simplePoisson(img, { time_window: tw, ratio, seed: 1 });
  const total = out.filter(Boolean).length;
  // each pixel spikes with prob ~ intensity*ratio per step -> ~ n*tw*0.25
  const expected = n * tw * 0.5 * 0.5;
  assert.ok(Math.abs(total - expected) / expected < 0.1, `spike count ${total} far from ${expected}`);
});

test('Poisson: zero-intensity pixels never spike', () => {
  const img = [0.0, 0.5];
  const out = M.poisson(img, { time_window: 50, ratio: 0.5, seed: 3 });
  assert.ok(out.every((x, i) => i % 2 === 0 ? !x : true), 'zero-intensity pixel must not spike');
});

test('Poisson: higher intensity gives more spikes on average', () => {
  const n = 200;
  const low = new Array(n).fill(0.1);
  const high = new Array(n).fill(0.9);
  const tw = 100;
  const count = (arr) => arr.filter(Boolean).length;
  const cLow = count(M.poisson(low, { time_window: tw, ratio: 0.5, seed: 5 }));
  const cHigh = count(M.poisson(high, { time_window: tw, ratio: 0.5, seed: 5 }));
  assert.ok(cHigh > cLow, 'higher intensity should spike more');
});

test('encoders reject a non-integer time_window', () => {
  assert.throws(() => M.simplePoisson([0.5], { time_window: 2.5, ratio: 0.5 }), /time_window/);
  assert.throws(() => M.poisson([0.5], { time_window: 0, ratio: 0.5 }), /time_window/);
  assert.throws(() => M.intensity2Latency([0.5], { time_window: -1 }), /time_window/);
});

// --- dendritic inputs (all compared against the golden) ---

test('SimpleDendriticInput matches the golden dense current', () => {
  const c = golden.connections.dense;
  const I = M.simpleDendriticInput({ pre_spike: c.spikes, weights: c.weights, nDst: 3 });
  assert.ok(closeArr(I, c.current));
});

test('One2OneDendriticInput matches the golden current', () => {
  const c = golden.connections.one2one;
  const I = M.one2oneDendriticInput({ pre_spike: c.spikes, weights: c.weights });
  assert.ok(closeArr(I, c.current));
});

test('Conv2dDendriticInput matches the golden current', () => {
  const c = golden.connections.conv2d;
  const I = M.conv2dDendriticInput({ pre_spike: c.spikes, srcH: 4, srcW: 4, weights: c.weights, kh: 3, kw: 3, stride: 1, padding: 0 });
  assert.ok(closeArr(I, c.current));
});

test('Local2dDendriticInput matches the golden current', () => {
  const c = golden.connections.local2d;
  const I = M.local2dDendriticInput({ pre_spike: c.spikes, srcH: 4, srcW: 4, weights: c.weights, kh: 3, kw: 3, stride: 1, padding: 0 });
  assert.ok(closeArr(I, c.current));
});

test('LateralDendriticInput matches the golden current', () => {
  const c = golden.connections.lateral;
  const I = M.lateralDendriticInput({ pre_spike: c.spikes, srcH: 4, srcW: 4, weights: c.weights, kh: 3, kw: 3 });
  assert.ok(closeArr(I, c.current));
});

test('AveragePool2D matches the golden current', () => {
  const c = golden.connections.avgpool;
  const I = M.averagePool2D({ pre_spike: c.spikes, srcH: 4, srcW: 4, dstH: 2, dstW: 2 });
  assert.ok(closeArr(I, c.current));
});

test('inhibitory source flips the sign of the current', () => {
  const c = golden.connections.dense;
  const exc = M.simpleDendriticInput({ pre_spike: c.spikes, weights: c.weights, nDst: 3, inhibitory: false });
  const inh = M.simpleDendriticInput({ pre_spike: c.spikes, weights: c.weights, nDst: 3, inhibitory: true });
  assert.ok(exc.every((x, i) => close(x, -inh[i])), 'inhibitory current must be the negation');
});

test('dendritic inputs reject a mismatched weight length', () => {
  assert.throws(() => M.simpleDendriticInput({ pre_spike: [1, 0], weights: [1, 2], nDst: 3 }), /weights length/);
  assert.throws(() => M.one2oneDendriticInput({ pre_spike: [1, 0], weights: [1] }), /weights length/);
});

// --- learning rule edge cases ---

test('STDP: no activity gives zero weight change', () => {
  const dw = M.stdpDw({
    weights: [0.5, 0.5, 0.5, 0.5],
    pre_spike: [0, 0], post_spike: [0, 0], pre_trace: [0, 0], post_trace: [0, 0],
    a_plus: 0.01, a_minus: 0.002,
  });
  assert.ok(dw.every((x) => x === 0));
});

test('STDP: pre-before-post potentiates, post-before-pre depresses', () => {
  // pre trace high, post spikes now -> LTP
  const ltp = M.stdpDw({
    weights: [0.5], pre_spike: [0], post_spike: [1], pre_trace: [1], post_trace: [0],
    a_plus: 0.01, a_minus: 0.002,
  })[0];
  // pre spikes now, post trace high -> LTD
  const ltd = M.stdpDw({
    weights: [0.5], pre_spike: [1], post_spike: [0], pre_trace: [0], post_trace: [1],
    a_plus: 0.01, a_minus: 0.002,
  })[0];
  assert.ok(ltp > 0, 'pre-before-post should potentiate');
  assert.ok(ltd < 0, 'post-before-pre should depress');
});

test('STDP: hard_bound blocks learning at the weight bounds', () => {
  const atMax = M.stdpDw({
    weights: [1.0], pre_spike: [0], post_spike: [1], pre_trace: [1], post_trace: [0],
    a_plus: 0.01, a_minus: 0.002, w_min: 0, w_max: 1, positive_bound: 'hard_bound',
  })[0];
  assert.ok(atMax === 0, 'no potentiation allowed at w_max with hard_bound');
});

test('STDP: soft_bound scales the update by (w-w_min)(w_max-w)', () => {
  const dw = M.stdpDw({
    weights: [0.5], pre_spike: [0], post_spike: [1], pre_trace: [1], post_trace: [0],
    a_plus: 0.01, a_minus: 0.002, w_min: 0, w_max: 1, positive_bound: 'soft_bound',
  })[0];
  assert.ok(close(dw, 1 * 1 * 0.01 * 0.25), 'soft_bound factor (0.5)(0.5)=0.25');
});

test('STDP rejects an unknown bound name', () => {
  assert.throws(
    () => M.stdpDw({ weights: [0.5], pre_spike: [0], post_spike: [1], pre_trace: [1], post_trace: [0], a_plus: 1, a_minus: 1, positive_bound: 'bogus' }),
    /unknown bound/,
  );
});

test('iSTDP: alpha is 2*rho*tau_s/1000 when alpha is not given', () => {
  assert.ok(close(M.istdpAlpha({ rho: 0.05, tau_s: 10 }), 0.001));
  assert.ok(close(M.istdpAlpha({ rho: 0.05, alpha: 0.02, tau_s: 10 }), 0.02));
});

test('iSTDP: inhibitory sign flips the pre-spike term', () => {
  const inh = M.istdpDw({
    weights: [0.5], pre_spike: [1], post_spike: [0], pre_trace: [0], post_trace: [0.5],
    rho: 0.05, lr: 1e-5, is_inhibitory: true, tau_s: 10,
  })[0];
  const exc = M.istdpDw({
    weights: [0.5], pre_spike: [1], post_spike: [0], pre_trace: [0], post_trace: [0.5],
    rho: 0.05, lr: 1e-5, is_inhibitory: false, tau_s: 10,
  })[0];
  assert.ok(inh !== exc, 'inhibitory and excitatory signs must differ');
});

test('stepTrace rejects a non-positive tau_s', () => {
  assert.throws(() => M.stepTrace(0, 1, { tau_s: 0, dt: 1 }), /tau_s/);
});

test('dopamineStep rejects a non-positive tau_dopamine', () => {
  assert.throws(() => M.dopamineStep({ dopamine_concentration: 0, tau_dopamine: 0, payoff: 1 }), /tau_dopamine/);
});

test('lifStep: spikes and resets when v crosses threshold', () => {
  // v += ((v_rest - v) + R*I) * dt/tau = 0.9 + (0 - 0.9 + 5*0.5)*0.1 = 1.01 >= 1.0
  const r = M.lifStep({ v: 0.9, I: 0.5, R: 5, threshold: 1.0, tau: 10, v_reset: 0, v_rest: 0, dt: 1 });
  assert.ok(r.spike === true);
  assert.ok(close(r.v, 0));
  // below threshold: no spike, v integrates
  const r2 = M.lifStep({ v: 0.0, I: 0.1, R: 5, threshold: 1.0, tau: 10, v_reset: 0, v_rest: 0, dt: 1 });
  assert.ok(r2.spike === false);
  assert.ok(close(r2.v, 0.05));
});

test('axonGetSpike rejects an out-of-range delay', () => {
  assert.throws(() => M.axonGetSpike([1, 0], 2), /delay/);
});
