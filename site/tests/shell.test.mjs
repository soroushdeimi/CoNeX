// node --test tests of the pure helpers in site/assets/js/core.js.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  encodeHash, decodeHash, rng, computeTicks, niceNum, dataRange, fmtTick,
  formatValue, listenerList, clampTime, playerAdvance,
} from '../assets/js/core.js';

// --- URL-hash encode/decode ------------------------------------------------

test('encodeHash builds key=value pairs joined by &', () => {
  assert.equal(encodeHash({ tau: 10, R: 1 }), 'tau=10&R=1');
});

test('encodeHash percent-encodes keys and values', () => {
  assert.equal(encodeHash({ 'a b': 'c&d' }), 'a%20b=c%26d');
});

test('decodeHash parses a hash with a leading #', () => {
  assert.deepEqual(decodeHash('#tau=10&R=1'), { tau: '10', R: '1' });
});

test('decodeHash parses a hash without a leading #', () => {
  assert.deepEqual(decodeHash('tau=10&R=1'), { tau: '10', R: '1' });
});

test('decodeHash round-trips encodeHash', () => {
  const params = { tau: '10', R: '1.5', name: 'a b&c' };
  assert.deepEqual(decodeHash(encodeHash(params)), params);
});

test('decodeHash skips malformed pairs instead of throwing', () => {
  // no '=', empty key, and a bare token are all dropped
  assert.deepEqual(decodeHash('good=1&=2&noequals&'), { good: '1' });
});

test('decodeHash returns {} for empty/undefined input', () => {
  assert.deepEqual(decodeHash(''), {});
  assert.deepEqual(decodeHash(undefined), {});
  assert.deepEqual(decodeHash('#'), {});
});

test('decodeHash decodes percent-encoded values', () => {
  assert.deepEqual(decodeHash('a%20b=c%26d'), { 'a b': 'c&d' });
});

// --- rng -------------------------------------------------------------------

test('rng is deterministic for a given seed', () => {
  const a = rng(42), b = rng(42);
  for (let i = 0; i < 5; i++) assert.equal(a.next(), b.next());
});

test('rng.next() stays in [0,1)', () => {
  const r = rng(7);
  for (let i = 0; i < 1000; i++) {
    const v = r.next();
    assert.ok(v >= 0 && v < 1, `next() out of range: ${v}`);
  }
});

test('rng with different seeds gives different streams', () => {
  assert.notEqual(rng(1).next(), rng(2).next());
});

test('rng.normal() returns finite values with mean near 0', () => {
  const r = rng(123);
  let sum = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) {
    const v = r.normal();
    assert.ok(Number.isFinite(v));
    sum += v;
  }
  assert.ok(Math.abs(sum / n) < 0.05, `mean too far from 0: ${sum / n}`);
});

// --- tick computation ------------------------------------------------------

test('computeTicks returns a single tick when min === max', () => {
  assert.deepEqual(computeTicks(5, 5), { ticks: [5], step: 1 });
});

test('computeTicks handles a negative value range', () => {
  const { ticks, step } = computeTicks(-10, 0);
  assert.ok(ticks[0] <= -10 + 1e-9, 'first tick below range');
  assert.ok(ticks[ticks.length - 1] >= 0 - 1e-9, 'last tick at/above range');
  assert.ok(step > 0, 'step must be positive');
  // ticks are monotonically increasing
  for (let i = 1; i < ticks.length; i++) assert.ok(ticks[i] > ticks[i - 1]);
});

test('computeTicks swaps min > max', () => {
  const { ticks } = computeTicks(10, 0);
  assert.ok(ticks[0] <= 0 + 1e-9);
  assert.ok(ticks[ticks.length - 1] >= 10 - 1e-9);
});

test('computeTicks includes 0 for a range crossing it', () => {
  const { ticks } = computeTicks(-5, 5);
  assert.ok(ticks.some((t) => t === 0), 'expected a 0 tick');
});

test('computeTicks returns empty ticks for non-finite input', () => {
  assert.deepEqual(computeTicks(NaN, 5), { ticks: [], step: 1 });
});

test('computeTicks ticks stay within [min,max]', () => {
  const { ticks } = computeTicks(0, 10);
  for (const t of ticks) assert.ok(t >= -1e-9 && t <= 10 + 1e-9, `tick ${t} out of range`);
});

// --- niceNum ---------------------------------------------------------------

test('niceNum rounds to 1, 2 or 5 times a power of ten', () => {
  assert.equal(niceNum(1.4, true), 1);
  assert.equal(niceNum(2.4, true), 2);
  assert.equal(niceNum(4.4, true), 5);
  assert.equal(niceNum(8.4, true), 10);
});

// --- dataRange -------------------------------------------------------------

test('dataRange does not pad below 0 when the data min is 0', () => {
  const [lo, hi] = dataRange(0, 10);
  assert.equal(lo, 0);
  assert.ok(hi >= 10 - 1e-9, 'range must cover the data max');
});

test('dataRange covers the data and pads outward by at most one step', () => {
  const [lo, hi] = dataRange(-3, 3);
  assert.ok(lo <= -3 + 1e-9);
  assert.ok(hi >= 3 - 1e-9);
});

test('dataRange snaps an exact step boundary instead of padding it', () => {
  assert.deepEqual(dataRange(0, 10), [0, 10]);
  assert.deepEqual(dataRange(0, 100), [0, 100]);
});

test('dataRange handles min === max', () => {
  assert.deepEqual(dataRange(5, 5), [4, 6]);
});

test('dataRange falls back to [0,1] for non-finite input', () => {
  assert.deepEqual(dataRange(NaN, 5), [0, 1]);
});

// --- fmtTick ---------------------------------------------------------------

test('fmtTick prints whole numbers without dropping digits', () => {
  assert.equal(fmtTick(10, 5), '10');
  assert.equal(fmtTick(100, 25), '100');
});

test('fmtTick prints decimals with the right precision', () => {
  assert.equal(fmtTick(0.1, 0.1), '0.1');
  assert.equal(fmtTick(-0.5, 0.5), '-0.5');
});

test('fmtTick handles very small and very large values', () => {
  assert.equal(fmtTick(1e-4, 1e-4), '0.0001');
  assert.equal(fmtTick(2.5e6, 5e5), '2500000');
});

test('fmtTick prints zero as "0"', () => {
  assert.equal(fmtTick(0, 1), '0');
});

// --- formatValue (slider/number readouts) ----------------------------------

test('formatValue caps a long float at 4 significant digits', () => {
  // large value with a fine step: the step alone would give 7 decimals,
  // the 4-significant-digit cap rounds it to 1235
  assert.equal(formatValue(1234.5678, 0.001), '1235');
  assert.equal(formatValue(123456.789, 0.01), '123457');
});

test('formatValue trims a repeating decimal to the step precision', () => {
  assert.equal(formatValue(0.16666666666666666, 0.001), '0.1667');
  assert.equal(formatValue(1 / 6, 0.01), '0.167');
});

test('formatValue keeps whole numbers and simple decimals intact', () => {
  assert.equal(formatValue(10, 1), '10');
  assert.equal(formatValue(100, 10), '100');
  assert.equal(formatValue(0.1, 0.1), '0.1');
  assert.equal(formatValue(-0.5, 0.5), '-0.5');
});

test('formatValue does not drop digits from large whole numbers', () => {
  assert.equal(formatValue(2500000, 500000), '2500000');
  assert.equal(formatValue(12345.6789, 100), '12346');
});

test('formatValue prints zero as "0"', () => {
  assert.equal(formatValue(0, 0.1), '0');
});

// --- listenerList (backing ui.slider/number/select/toggle .onInput) --------

test('listenerList.emit calls each listener in order with the value', () => {
  const ll = listenerList();
  const seen = [];
  ll.add((v) => seen.push(['a', v]));
  ll.add((v) => seen.push(['b', v]));
  ll.emit(5);
  assert.deepEqual(seen, [['a', 5], ['b', 5]]);
});

test('listenerList.add returns an unsubscribe that stops that listener', () => {
  const ll = listenerList();
  const seen = [];
  const off = ll.add((v) => seen.push(['a', v]));
  ll.add((v) => seen.push(['b', v]));
  off();
  ll.emit(1);
  assert.deepEqual(seen, [['b', 1]]);
});

test('listenerList.size tracks the number of listeners', () => {
  const ll = listenerList();
  assert.equal(ll.size(), 0);
  const off = ll.add(() => {});
  assert.equal(ll.size(), 1);
  off();
  assert.equal(ll.size(), 0);
});

test('listenerList.emit is safe with no listeners and after unsubscribe', () => {
  const ll = listenerList();
  ll.emit(3); // no listeners: no throw
  const off = ll.add(() => {});
  off();
  ll.emit(4); // all unsubscribed: no throw
});

// --- clampTime / playerAdvance (backing ui.player) -------------------------

test('clampTime keeps t within [0, duration]', () => {
  assert.equal(clampTime(5, 100), 5);
  assert.equal(clampTime(-5, 100), 0);
  assert.equal(clampTime(150, 100), 100);
});

test('clampTime maps non-finite input to 0', () => {
  assert.equal(clampTime(NaN, 100), 0);
  assert.equal(clampTime(Infinity, 100), 0);
  assert.equal(clampTime(-Infinity, 100), 0);
});

test('playerAdvance advances at speed * duration / playSeconds per real second', () => {
  // duration 100 over 5 real seconds at 1x => 20 model units per real second
  assert.equal(playerAdvance(0, 1000, 100, 5, 1, false), 20);
  assert.equal(playerAdvance(0, 500, 100, 5, 1, false), 10);
  assert.equal(playerAdvance(10, 500, 100, 5, 1, false), 20);
});

test('playerAdvance scales with speed', () => {
  assert.equal(playerAdvance(0, 1000, 100, 5, 2, false), 40);
  assert.equal(playerAdvance(0, 1000, 100, 5, 0.5, false), 10);
});

test('playerAdvance clamps at duration when not looping', () => {
  assert.equal(playerAdvance(90, 1000, 100, 5, 1, false), 100);
  assert.equal(playerAdvance(0, 10000, 100, 5, 1, false), 100);
});

test('playerAdvance wraps past the end when looping', () => {
  assert.equal(playerAdvance(90, 1000, 100, 5, 1, true), 10);
  assert.equal(playerAdvance(95, 2000, 100, 5, 1, true), 35);
  // wraps more than once in a single large step
  assert.equal(playerAdvance(95, 6000, 100, 5, 1, true), 15);
});

test('playerAdvance with a zero step stays put', () => {
  assert.equal(playerAdvance(30, 0, 100, 5, 1, false), 30);
  assert.equal(playerAdvance(30, 0, 100, 5, 1, true), 30);
});
