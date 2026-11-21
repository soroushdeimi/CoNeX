// Tests for site/assets/js/anim.js: the pure model-state -> drawn-quantity
// mappings used by the hero animations. Values are hand-computed.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as A from '../assets/js/anim.js';

// --- syn-stdp ---

test('stdpHalo clamps the trace to [0,1]', () => {
  assert.equal(A.stdpHalo(0), 0);
  assert.equal(A.stdpHalo(0.42), 0.42);
  assert.equal(A.stdpHalo(1), 1);
  assert.equal(A.stdpHalo(1.3), 1); // a trace can overshoot 1; clamp
  assert.equal(A.stdpHalo(-0.2), 0);
});

test('stdpSynapsePx maps the weight over [wMin,wMax] to [minPx,maxPx]', () => {
  assert.equal(A.stdpSynapsePx(0, 0, 1), 3);
  assert.equal(A.stdpSynapsePx(1, 0, 1), 17);
  assert.equal(A.stdpSynapsePx(0.5, 0, 1), 10);
  assert.equal(A.stdpSynapsePx(0.25, 0, 1), 6.5);
  assert.equal(A.stdpSynapsePx(2, 0, 1), 17); // clamp above wMax
  assert.equal(A.stdpSynapsePx(-1, 0, 1), 3); // clamp below wMin
});

test('stdpHighlightDelta picks the causal side for each spike', () => {
  // post spike 4 ms after the last pre -> +4 (potentiation side)
  assert.equal(A.stdpHighlightDelta(14, 10, 0, 0, 1), 4);
  // pre spike 3 ms after the last post -> -3 (depression side)
  assert.equal(A.stdpHighlightDelta(13, 0, 10, 1, 0), -3);
  // both fire: the post spike (potentiation) is reported
  assert.equal(A.stdpHighlightDelta(14, 10, 10, 1, 1), 4);
  // no pairing yet -> null
  assert.equal(A.stdpHighlightDelta(1, -1, -1, 1, 0), null);
  assert.equal(A.stdpHighlightDelta(1, -1, -1, 0, 1), null);
});

test('stdpSpikePulse is in flight only between the spike and the synapse', () => {
  // before the spike: not in flight
  assert.equal(A.stdpSpikePulse(4, 10, 15), null);
  // at the spike: just leaving the neuron
  assert.equal(A.stdpSpikePulse(10, 10, 15), 0);
  // halfway: 0.5
  assert.equal(A.stdpSpikePulse(17.5, 10, 15), 0.5);
  // just before arrival: close to 1
  assert.ok(A.stdpSpikePulse(24.9, 10, 15) > 0.99 && A.stdpSpikePulse(24.9, 10, 15) < 1);
  // at/after arrival: gone
  assert.equal(A.stdpSpikePulse(25, 10, 15), null);
  // no spike yet (t0 < 0): never in flight
  assert.equal(A.stdpSpikePulse(5, -1, 15), null);
  // non-positive travel: never in flight
  assert.equal(A.stdpSpikePulse(5, 0, 0), null);
});

test('stdpNarration names the pairing and the sign of dw', () => {
  assert.equal(A.stdpNarration({ preSpike: 1, postSpike: 1, dw: 0 }), 'pre and post fire together: potentiation and depression both apply');
  assert.equal(A.stdpNarration({ preSpike: 1, postSpike: 0, dw: 0 }), 'pre spike: pre_trace rises; it will potentiate if the post fires next');
  assert.equal(A.stdpNarration({ preSpike: 0, postSpike: 1, dw: 0 }), 'post spike: post_trace rises; it depresses if the pre fired before');
  assert.equal(A.stdpNarration({ preSpike: 0, postSpike: 0, dw: 0.001 }), 'pre before post: potentiation (dw > 0)');
  assert.equal(A.stdpNarration({ preSpike: 0, postSpike: 0, dw: -0.001 }), 'post before pre: depression (dw < 0)');
  assert.equal(A.stdpNarration({ preSpike: 0, postSpike: 0, dw: 0 }), 'traces decay: no spike pairing this step');
});

// --- syn-istdp ---

test('istdpMeterPos maps a rate over [0,maxRate] to [0,1]', () => {
  assert.equal(A.istdpMeterPos(0, 60), 0);
  assert.equal(A.istdpMeterPos(30, 60), 0.5);
  assert.equal(A.istdpMeterPos(60, 60), 1);
  assert.equal(A.istdpMeterPos(90, 60), 1); // clamp
  assert.equal(A.istdpMeterPos(30, 0), 0); // degenerate max
});

test('istdpSynapsePx maps the weight over [0,wMax] to [minPx,maxPx]', () => {
  assert.equal(A.istdpSynapsePx(0, 1), 2);
  assert.equal(A.istdpSynapsePx(1, 1), 14);
  assert.equal(A.istdpSynapsePx(0.5, 1), 8);
  assert.equal(A.istdpSynapsePx(2, 1), 14); // clamp
});

test('istdpNarration reports the direction of the weight change', () => {
  assert.equal(A.istdpNarration(30, 20), 'post rate above target: inhibitory weight grows to suppress it');
  assert.equal(A.istdpNarration(10, 20), 'post rate below target: inhibitory weight shrinks');
  assert.equal(A.istdpNarration(20, 20), 'post rate at target rho: the weight is stable');
  // 5% band: 21 is within 5% of 20 -> stable
  assert.equal(A.istdpNarration(21, 20), 'post rate at target rho: the weight is stable');
});

// --- syn-rstdp ---

test('rstdpEligGlow scales |c| by its max and clamps', () => {
  assert.equal(A.rstdpEligGlow(0, 1), 0);
  assert.equal(A.rstdpEligGlow(0.5, 1), 0.5);
  assert.equal(A.rstdpEligGlow(-0.5, 1), 0.5); // sign-insensitive
  assert.equal(A.rstdpEligGlow(2, 1), 1); // clamp
  assert.equal(A.rstdpEligGlow(0.5, 0), 0); // degenerate max
});

test('rstdpDopGlow scales dopamine by its max and clamps', () => {
  assert.equal(A.rstdpDopGlow(0, 1), 0);
  assert.equal(A.rstdpDopGlow(0.25, 1), 0.25);
  assert.equal(A.rstdpDopGlow(1, 1), 1);
  assert.equal(A.rstdpDopGlow(3, 1), 1); // clamp
});

test('rstdpNarration reports the reward-window outcome', () => {
  assert.equal(A.rstdpNarration({ postSpike: 1, inWindow: true, dop: 0 }), 'post fires in the reward window: dopamine released, eligibility becomes weight change');
  assert.equal(A.rstdpNarration({ postSpike: 1, inWindow: false, dop: 0 }), 'post fires outside the window: no reward, eligibility decays');
  assert.equal(A.rstdpNarration({ postSpike: 0, inWindow: false, dop: 0.5 }), 'dopamine present: weight += eligibility * dopamine');
  assert.equal(A.rstdpNarration({ postSpike: 0, inWindow: false, dop: 0 }), 'eligibility trace decays (tau_c); waiting for a reward');
});

// --- syn-connections ---

test('connProgress and connBarFill clamp to [0,1]', () => {
  assert.equal(A.connProgress(0), 0);
  assert.equal(A.connProgress(0.5), 0.5);
  assert.equal(A.connProgress(1), 1);
  assert.equal(A.connProgress(2), 1);
  assert.equal(A.connBarFill(0, 1), 0);
  assert.equal(A.connBarFill(0.5, 1), 0.5);
  assert.equal(A.connBarFill(-0.5, 1), 0.5); // magnitude
  assert.equal(A.connBarFill(2, 1), 1); // clamp
  assert.equal(A.connBarFill(0.5, 0), 0); // degenerate
});

// --- enc-poisson ---

test('encSparkProgress clamps to [0,1]', () => {
  assert.equal(A.encSparkProgress(0), 0);
  assert.equal(A.encSparkProgress(0.7), 0.7);
  assert.equal(A.encSparkProgress(1.5), 1);
});

test('encNarration names the encoder code', () => {
  assert.match(A.encNarration('i2l'), /latency code/);
  assert.match(A.encNarration('poisson'), /intervals ~ Poisson/);
  assert.match(A.encNarration('simple'), /probability intensity \* ratio/);
});

// --- learn-stdp-family ---

test('stdpFamCellFrac maps a weight over [wMin,wMax] to [0,1]', () => {
  assert.equal(A.stdpFamCellFrac(0, 0, 1), 0);
  assert.equal(A.stdpFamCellFrac(0.5, 0, 1), 0.5);
  assert.equal(A.stdpFamCellFrac(1, 0, 1), 1);
  assert.equal(A.stdpFamCellFrac(2, 0, 1), 1); // clamp
  assert.equal(A.stdpFamCellFrac(0.5, 0.5, 0.5), 0.5); // degenerate range
});

test('stdpFamNarration names the rule and topology', () => {
  assert.match(A.stdpFamNarration('STDP', 'Simple'), /dense weight matrix/);
  assert.match(A.stdpFamNarration('STDP', 'Conv2d'), /shared kernel/);
  assert.match(A.stdpFamNarration('iSTDP', 'Simple'), /iSTDP/);
  assert.match(A.stdpFamNarration('RSTDP', 'Local2d'), /local kernel/);
});

// --- learn-bounds ---

test('boundsBeadPos maps a weight over [wMin,wMax] to [0,1]', () => {
  assert.equal(A.boundsBeadPos(0, 0, 1), 0);
  assert.equal(A.boundsBeadPos(0.5, 0, 1), 0.5);
  assert.equal(A.boundsBeadPos(1, 0, 1), 1);
  assert.equal(A.boundsBeadPos(2, 0, 1), 1); // clamp
  assert.equal(A.boundsBeadPos(-1, 0, 1), 0); // clamp
});

test('boundsSpring clamps the bound factor to [0,1]', () => {
  assert.equal(A.boundsSpring(0), 0);
  assert.equal(A.boundsSpring(0.25), 0.25);
  assert.equal(A.boundsSpring(1), 1);
  assert.equal(A.boundsSpring(2), 1); // soft_bound can exceed 1
});

test('boundsNarration reports the stuck state at a boundary', () => {
  assert.match(A.boundsNarration('hard_bound', 1, 0, 1, 'plus'), /stuck/);
  assert.match(A.boundsNarration('soft_bound', 0, 0, 1, 'minus'), /stop smoothly/);
  assert.match(A.boundsNarration('no_bound', 0.5, 0, 1, 'plus'), /potentiation pushes/);
  assert.match(A.boundsNarration('no_bound', 0.5, 0, 1, 'minus'), /depression pushes/);
});
