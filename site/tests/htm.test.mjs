import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  SDR,
  sdrConfigNActive,
  falseMatchProbability,
  expectedOverlap,
  encodeScalar,
  SDRClassifier,
  UnionSDR,
  SpatialPooler,
  SpatialPoolerDefaults,
  TemporalMemory,
  TemporalMemoryDefaults,
} from '../assets/js/models/htm.js';
import { sdrHeroState } from '../assets/plates/htm-sdr.js';
import { encoderHeroState } from '../assets/plates/htm-encoder.js';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, 'golden', 'htm.json'), 'utf8'));

// Tolerance for float32 golden values (CoNeX runs in float32).
const F32 = 1e-5;
function close(a, b, tol = F32) {
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} within ${tol} of ${b}`);
}
function closeArr(a, b, tol = F32) {
  assert.equal(a.length, b.length, `length ${a.length} != ${b.length}`);
  for (let i = 0; i < a.length; i++) close(a[i], b[i], tol);
}
function sameSet(a, b) {
  assert.deepEqual([...a].sort((x, y) => x - y), [...b].sort((x, y) => x - y));
}

// --- SDR -------------------------------------------------------------------

test('sdr: overlap, union, intersection, difference match golden', () => {
  const g = golden.sdr;
  const a = new SDR(g.size, g.a_indices);
  const b = new SDR(g.size, g.b_indices);
  const c = new SDR(g.size);
  assert.equal(a.n_active, g.n_active[0]);
  assert.equal(b.n_active, g.n_active[1]);
  assert.equal(c.n_active, g.n_active[2]);
  close(a.sparsity, g.sparsity[0]);
  assert.equal(a.overlap(b), g.overlap_ab);
  assert.equal(a.overlap(c), g.overlap_ac);
  close(a.overlapScore(b), g.overlap_score_ab);
  close(a.overlapScore(c), g.overlap_score_ac);
  close(a.jaccard(b), g.jaccard_ab);
  const u = a.union(b);
  assert.equal(u.n_active, g.union_n_active);
  sameSet(u.indices, g.union_indices);
  sameSet(a.intersection(b).indices, g.intersection_indices);
  sameSet(a.difference(b).indices, g.difference_indices);
  assert.equal(a.matchThreshold(b, 4), g.match_threshold_ab_4);
  assert.equal(a.matchThreshold(b, 5), g.match_threshold_ab_5);
});

test('sdr: empty SDR overlap and intersection are zero', () => {
  const a = new SDR(16, [1, 2, 3]);
  const e = new SDR(16);
  assert.equal(a.overlap(e), 0);
  assert.equal(a.overlapScore(e), 0);
  assert.equal(a.intersection(e).n_active, 0);
  assert.equal(a.difference(e).n_active, 3);
});

test('sdr: duplicate indices are collapsed', () => {
  const s = new SDR(16, [1, 1, 2, 2, 2, 3]);
  assert.equal(s.n_active, 3);
  assert.deepEqual(s.indices, [1, 2, 3]);
});

test('sdr: bad size throws with the value', () => {
  assert.throws(() => new SDR(-1), /-1/);
  assert.throws(() => new SDR(1.5), /1\.5/);
});

test('sdr: SDRConfig.n_active matches golden (max(1, int(size*sparsity)))', () => {
  assert.equal(sdrConfigNActive(2048, 0.02), golden.sdr.config_n_active[0]);
  assert.equal(sdrConfigNActive(100, 0.001), golden.sdr.config_n_active[1]);
});

// --- False-match probability ----------------------------------------------

test('false-match: hypergeometric tail matches golden', () => {
  for (const c of golden.false_match) {
    const p = falseMatchProbability(c.n, c.w, c.w, c.theta);
    // relative tolerance: tiny tails need a relative, not absolute, check
    assert.ok(
      Math.abs(p - c.p) <= 1e-9 * Math.max(1, c.p) || Math.abs(p - c.p) <= 1e-12,
      `n=${c.n} w=${c.w} theta=${c.theta}: js=${p} gold=${c.p}`
    );
  }
});

test('false-match: expected overlap is w^2/n', () => {
  for (const c of golden.false_match) {
    close(expectedOverlap(c.n, c.w), c.expected_overlap, 1e-12);
  }
});

test('false-match: theta=0 gives probability 1', () => {
  close(falseMatchProbability(64, 8, 8, 0), 1.0, 1e-12);
});

test('false-match: theta above w gives 0', () => {
  assert.equal(falseMatchProbability(64, 8, 8, 9), 0);
});

test('false-match: bad parameters throw', () => {
  assert.throws(() => falseMatchProbability(0, 8, 8, 4), /n=0/);
  assert.throws(() => falseMatchProbability(64, 8, 65, 4), /w=65/);
  assert.throws(() => falseMatchProbability(64, 65, 8, 4), /wx=65/);
  assert.throws(() => falseMatchProbability(64, 8, 8, -1), /-1/);
});

// --- SDREncoder (scalar) ---------------------------------------------------

test('encoder: scalar codes match golden indices', () => {
  const g = golden.encoder;
  const opts = { size: g.size, n_active: g.n_active, min_val: g.min_val, max_val: g.max_val };
  for (const [value, indices] of Object.entries(g.codes)) {
    const code = encodeScalar(parseFloat(value), opts);
    sameSet(code.indices, indices);
    assert.equal(code.n_active, g.n_active);
  }
});

test('encoder: overlap falls with distance between values', () => {
  const g = golden.encoder;
  const opts = { size: g.size, n_active: g.n_active, min_val: g.min_val, max_val: g.max_val };
  const e = (v) => encodeScalar(v, opts);
  assert.equal(e(0.5).overlap(e(0.52)), g['overlap_0.5_0.52']);
  assert.equal(e(0.5).overlap(e(0.0)), g['overlap_0.5_0.0']);
  assert.ok(e(0.5).overlap(e(0.52)) > e(0.5).overlap(e(0.0)));
});

test('encoder: values clamp to the configured range', () => {
  const g = golden.encoder;
  const opts = { size: g.size, n_active: g.n_active, min_val: g.min_val, max_val: g.max_val };
  assert.equal(encodeScalar(-5.0, opts).equals(encodeScalar(0.0, opts)), g.clamp_equal);
  assert.equal(encodeScalar(5.0, opts).equals(encodeScalar(1.0, opts)), g.clamp_equal);
});

// --- SDRClassifier ---------------------------------------------------------

test('classifier: learned patterns are recovered with golden scores', () => {
  const g = golden.classifier;
  const clf = new SDRClassifier({
    n_classes: g.n_classes,
    sdr_size: g.sdr_size,
    learning_rate: g.learning_rate,
  });
  const patterns = {
    0: new SDR(g.sdr_size, [0, 1, 2, 3, 4, 5, 6, 7]),
    1: new SDR(g.sdr_size, [20, 21, 22, 23, 24, 25, 26, 27]),
    2: new SDR(g.sdr_size, [40, 41, 42, 43, 44, 45, 46, 47]),
  };
  for (const [label, pat] of Object.entries(patterns)) {
    for (let i = 0; i < g.n_examples_per_class; i++) clf.learn(pat, parseInt(label));
  }
  const [pred0, scores0] = clf.infer(patterns[0]);
  const [pred1, scores1] = clf.infer(patterns[1]);
  assert.equal(pred0, g.predict_0);
  assert.equal(pred1, g.predict_1);
  closeArr(scores0, g.scores_0);
  closeArr(scores1, g.scores_1);
  closeArr(Array.from(clf.counts), g.class_counts);
});

test('classifier: untrained infer does not crash and returns a valid class', () => {
  const clf = new SDRClassifier({ n_classes: 3, sdr_size: 64 });
  const [pred, scores] = clf.infer(new SDR(64, [0, 1, 2, 3, 4, 5, 6, 7]));
  assert.ok(pred >= 0 && pred < 3);
  assert.equal(scores.length, 3);
});

test('classifier: out-of-range label throws with the value', () => {
  const clf = new SDRClassifier({ n_classes: 3, sdr_size: 64 });
  assert.throws(() => clf.learn(new SDR(64, [0, 1]), 7), /7/);
});

// --- UnionSDR --------------------------------------------------------------

// Build the SpatialPooler exactly as the golden script sets it up: identity
// potential map, column c connected to inputs 0..(7+c) at perm 0.1 else 0.05.
function buildSp(boostStrength) {
  const g = golden.spatial_pooler;
  const sp = new SpatialPooler({
    input_size: g.input_size,
    n_columns: g.n_columns,
    potential_pct: g.potential_pct,
    local_area_density: g.local_area_density,
    syn_perm_connected: g.syn_perm_connected,
    syn_perm_active_inc: g.syn_perm_active_inc,
    syn_perm_inactive_dec: g.syn_perm_inactive_dec,
    duty_cycle_period: g.duty_cycle_period,
    boost_strength: boostStrength,
  });
  const { n_columns, n_potential } = sp;
  const input_size = sp.cfg.input_size;
  sp.potential_inputs = Array.from({ length: n_columns }, () =>
    Array.from({ length: n_potential }, (_, i) => i)
  );
  sp.permanences = Array.from({ length: n_columns }, (_, c) =>
    Array.from({ length: input_size }, (_, i) => (i < 8 + c ? 0.1 : 0.05))
  );
  sp.boost_factors = new Array(n_columns).fill(1.0);
  sp.overlap_duty = new Array(n_columns).fill(0.0);
  sp.active_duty = new Array(n_columns).fill(0.0);
  return sp;
}

function inputBits(active) {
  const x = new Array(16).fill(0);
  for (const i of active) x[i] = 1;
  return x;
}

test('spatial-pooler: winners, permanences, duty match golden', () => {
  const g = golden.spatial_pooler;
  const sp = buildSp(0.0);
  assert.equal(sp.n_active, g.n_active);
  assert.equal(sp.n_potential, g.n_potential);
  const y = inputBits(g.input_active_bits);

  const a1 = sp.compute(y, true);
  sameSet(a1, g.active_step1);
  closeArr(sp.permanences[6], g.perm_col6_after_step1);
  close(sp.overlap_duty[6], g.duty_col6_after_step1[0]);
  close(sp.active_duty[6], g.duty_col6_after_step1[1]);

  const a2 = sp.compute(y, false);
  sameSet(a2, g.active_step2);

  const a3 = sp.compute(y, true);
  sameSet(a3, g.active_step3);
  closeArr(sp.permanences[6], g.perm_col6_after_step3);

  const aEmpty = sp.compute(new Array(16).fill(0), false);
  sameSet(aEmpty, g.active_empty);
});

test('spatial-pooler: boosting produces golden boost factors', () => {
  const g = golden.spatial_pooler;
  const sp = buildSp(1.0);
  const y = inputBits(g.input_active_bits);
  sp.compute(y, true);
  closeArr(sp.boost_factors, g.boost_factors_after_1, 1e-6);
});

test('spatial-pooler: empty input activates nothing', () => {
  const sp = buildSp(0.0);
  const a = sp.compute(new Array(16).fill(0), false);
  assert.equal(a.length, 0);
});

test('spatial-pooler: permanences stay in [0,1] after many steps', () => {
  const sp = buildSp(0.0);
  const y = inputBits([8, 9, 10, 11, 12, 13, 14, 15]);
  for (let i = 0; i < 50; i++) sp.compute(y, true);
  for (const row of sp.permanences) {
    for (const p of row) {
      assert.ok(p >= 0 && p <= 1, `permanence ${p} out of range`);
    }
  }
});

test('union: add/remove/contains match golden', () => {
  const g = golden.union;
  const a = new SDR(g.size, [0, 1, 2, 3, 4, 5, 6, 7]);
  const b = new SDR(g.size, [4, 5, 6, 7, 8, 9, 10, 11]);
  const c = new SDR(g.size, [60, 61, 62, 63, 64, 65, 66, 67]);
  const u = new UnionSDR(g.size);
  u.add(a);
  u.add(b);
  u.add(c);
  assert.equal(u.n_active, g.n_active_after_3);
  close(u.sparsity, g.sparsity_after_3);
  assert.equal(u.n_members, g.n_members_after_3);
  assert.equal(u.contains(a, 0.5), g.contains_a);
  u.remove(a);
  assert.equal(u.n_active, g.n_active_after_remove_a);
  sameSet(u.toSDR().indices, g.union_indices_after_remove_a);
});

// --- TemporalMemory --------------------------------------------------------

function buildTm() {
  const g = golden.temporal_memory;
  return new TemporalMemory({
    n_columns: g.n_columns,
    cells_per_column: g.cells_per_column,
    activation_threshold: g.activation_threshold,
    learning_threshold: g.learning_threshold,
    max_new_synapses: g.max_new_synapses,
    permanence_increment: g.permanence_increment,
    permanence_decrement: g.permanence_decrement,
  });
}

test('temporal-memory: replay after learning matches golden', () => {
  const g = golden.temporal_memory;
  const tm = buildTm();
  const S = g.symbols;
  const seq1 = [S.A, S.B, S.C, S.D];
  const seq2 = [S.X, S.B, S.C, S.Y];
  for (let e = 0; e < 8; e++) {
    for (const step of [...seq1, ...seq2]) tm.compute(step, true);
  }
  // segment counts per cell
  const segCounts = [];
  for (let c = 0; c < tm.n_cells; c++) segCounts.push(tm.segments.countSegments(c));
  assert.deepEqual(segCounts, g.segment_counts);

  // replay without learning
  for (let i = 0; i < seq1.length + seq2.length; i++) {
    const step = i < 4 ? seq1[i] : seq2[i - 4];
    tm.compute(step, false);
    const r = g.replay[i];
    sameSet(tm.active_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), r.active_cells);
    sameSet(tm.winner_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), r.winner_cells);
    sameSet(tm.predictive_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), r.predictive_cells);
    sameSet(tm.bursting_columns.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), r.bursting_columns);
    close(tm.anomaly_score, r.anomaly, 1e-9);
  }
});

test('temporal-memory: context decides D vs Y after the shared C', () => {
  const g = golden.temporal_memory;
  const tm = buildTm();
  const S = g.symbols;
  const seq1 = [S.A, S.B, S.C, S.D];
  const seq2 = [S.X, S.B, S.C, S.Y];
  for (let e = 0; e < 8; e++) for (const step of [...seq1, ...seq2]) tm.compute(step, true);

  // After A-B-C, the prediction should include D's columns (12,13,14).
  tm.compute(S.A, false);
  tm.compute(S.B, false);
  tm.compute(S.C, false);
  const predAfterC = tm.predictive_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0);
  // D's cells are 12*4..15*4 = 48..59; Y's cells are 15*4..17*4 = 60..69.
  const predictsD = predAfterC.some((c) => c >= 48 && c < 60);
  const predictsY = predAfterC.some((c) => c >= 60 && c < 72);
  assert.ok(predictsD, 'after A-B-C, D should be predicted');
  assert.ok(predictsY, 'after A-B-C (shared C), Y should also be predicted');

  // After X-B-C, the prediction should include Y's columns (60..69).
  tm.compute(S.X, false);
  tm.compute(S.B, false);
  tm.compute(S.C, false);
  const predAfterC2 = tm.predictive_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0);
  assert.ok(predAfterC2.some((c) => c >= 60 && c < 72), 'after X-B-C, Y should be predicted');
});

test('temporal-memory: novel input bursts with anomaly 1.0', () => {
  const g = golden.temporal_memory;
  const tm = buildTm();
  tm.compute(g.symbols.A, false);
  sameSet(tm.bursting_columns.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), g.fresh.bursting_columns);
  sameSet(tm.winner_cells.map((a, j) => (a ? j : -1)).filter((j) => j >= 0), g.fresh.winner_cells);
  close(tm.anomaly_score, g.fresh.anomaly, 1e-9);
});

test('temporal-memory: one winner cell per active column on novel input', () => {
  const tm = buildTm();
  tm.compute([0, 1, 2], false);
  const winners = tm.winner_cells.filter(Boolean).length;
  assert.equal(winners, 3);
});

test('temporal-memory: empty input gives anomaly 0 and no active cells', () => {
  const tm = buildTm();
  tm.compute([], false);
  assert.equal(tm.anomaly_score, 0);
  assert.equal(tm.active_cells.filter(Boolean).length, 0);
});

test('temporal-memory: learning decrements permanence on inactive targets', () => {
  const g = golden.tm_decrement;
  const tm = new TemporalMemory({
    n_columns: g.n_columns,
    cells_per_column: g.cells_per_column,
    activation_threshold: g.activation_threshold,
    learning_threshold: g.learning_threshold,
    max_new_synapses: g.max_new_synapses,
    permanence_increment: g.permanence_increment,
    permanence_decrement: g.permanence_decrement,
  });
  const P2 = [0, 1];
  const P1 = [0];
  const W = [2];
  for (const step of [P2, W, P1, W]) tm.compute(step, true);

  assert.equal(tm.segments.countSegments(g.winner_cell), g.n_segments);
  const seg = tm.segments.segments[g.winner_cell][0];
  // permanences keyed by target cell (order-independent)
  const permsByTarget = {};
  for (const s of seg) permsByTarget[s.target] = s.permanence;
  for (const [target, expected] of Object.entries(g.perms_by_target)) {
    close(permsByTarget[parseInt(target)], expected, 1e-5);
  }
  // The inactive target must be below initial permanence (decremented), the
  // active target above it (incremented).
  assert.ok(permsByTarget[2] < 0.21, 'inactive target permanence should be decremented');
  assert.ok(permsByTarget[0] > 0.21, 'active target permanence should be incremented');
});

// --- Hero animation mapping (AMENDMENT 5) ----------------------------------

// A tiny fixed run: n=4, two samples, fixed code a = {0,1}.
function sdrRun() {
  return {
    n: 4,
    m: 2,
    w: 2,
    theta: 2,
    a: new SDR(4, [0, 1]),
    bits: [[1, 1, 0, 0], [0, 1, 1, 0]], // sample 0 overlaps a in 2 bits, sample 1 in 1
    ov: [2, 1],
    pExp: 1.0,
  };
}

test('sdr hero: at an integer step the sample is that sample, not a blend', () => {
  const st = sdrHeroState(sdrRun(), 0);
  assert.equal(st.idx, 0);
  assert.equal(st.f, 0);
  assert.deepEqual(st.sample, [1, 1, 0, 0]);
  assert.equal(st.overlapBar, 2);
});

test('sdr hero: at the midpoint the sample is a 50/50 blend and the bar eases', () => {
  const st = sdrHeroState(sdrRun(), 0.5);
  assert.equal(st.idx, 0);
  assert.equal(st.next, 1);
  assert.equal(st.f, 0.5);
  // bit 0: 1->0, bit 1: 1->1, bit 2: 0->1, bit 3: 0->0
  assert.deepEqual(st.sample, [0.5, 1, 0.5, 0]);
  // overlap bar eases between 2 and 1 at f=0.5 -> 1.5
  assert.equal(st.overlapBar, 1.5);
});

test('sdr hero: the overlap mask marks only bits shared with the fixed code', () => {
  const st = sdrHeroState(sdrRun(), 0);
  // a = {0,1}; sample 0 = {0,1}; intersection = {0,1}
  assert.deepEqual(st.overlapMask, [1, 1, 0, 0]);
});

test('sdr hero: narration names the sample, overlap and the match threshold', () => {
  const st = sdrHeroState(sdrRun(), 0);
  assert.match(st.narration, /sample 1\/2/);
  assert.match(st.narration, /overlap 2 bits/);
  assert.match(st.narration, /a match/);
  const st1 = sdrHeroState(sdrRun(), 1);
  assert.match(st1.narration, /overlap 1 bits/);
  assert.doesNotMatch(st1.narration, /a match/);
});

test('sdr hero: time is clamped to the last sample at the end of the run', () => {
  const st = sdrHeroState(sdrRun(), 99);
  assert.equal(st.idx, 1);
  assert.equal(st.next, 1);
  assert.deepEqual(st.sample, [0, 1, 1, 0]);
});

// Encoder hero: size=10, n_active=4, range [0,1]. center = trunc(norm*6).
// v=0 -> bits 0-3; v=0.5 -> bits 3-6; v=1 -> bits 6-9. Reference at 0.5.
function encRun() {
  const o = { size: 10, n_active: 4, min_val: 0, max_val: 1 };
  return { o, ref: 0.5, base: encodeScalar(0.5, o) };
}

test('encoder hero: at the reference value the sweep block fully overlaps', () => {
  const st = encoderHeroState(encRun(), 0.5);
  assert.equal(st.v, 0.5);
  assert.equal(st.norm, 0.5);
  assert.equal(st.nOverlap, 4); // same block as the reference
  assert.deepEqual(st.curDense, st.refDense);
});

test('encoder hero: at the range edge the block shares one bit with the reference', () => {
  const st = encoderHeroState(encRun(), 0); // v=0 -> bits 0-3, ref bits 3-6
  assert.equal(st.nOverlap, 1); // only bit 3
  assert.equal(st.overlap[3], 1);
  assert.equal(st.overlap[0], 0);
});

test('encoder hero: the needle position is the normalized value', () => {
  assert.equal(encoderHeroState(encRun(), 0).norm, 0);
  assert.equal(encoderHeroState(encRun(), 1).norm, 1);
});

test('encoder hero: narration names the value, block size and overlap', () => {
  const st = encoderHeroState(encRun(), 0.5);
  assert.match(st.narration, /value 0\.50 a\.u\./);
  assert.match(st.narration, /block of 4 bits/);
  assert.match(st.narration, /overlap with reference \(0\.50\) = 4 bits/);
});
