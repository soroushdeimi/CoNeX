// Pure helpers shared by ui.js and the node tests. No DOM, no browser APIs.

// Encode a flat object as a URL hash body: {tau: 10, R: 1} -> "tau=10&R=1".
export function encodeHash(params) {
  const parts = [];
  for (const [k, v] of Object.entries(params)) {
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return parts.join('&');
}

// Parse a URL hash (with or without leading #) into a flat object of strings.
// Malformed pairs (no '=', empty key) are skipped, never thrown on.
export function decodeHash(hash) {
  const out = {};
  if (hash === undefined || hash === null) return out;
  const s = String(hash).replace(/^[#?]/, '');
  for (const pair of s.split('&')) {
    if (!pair) continue;
    const eq = pair.indexOf('=');
    if (eq <= 0) continue;
    let k, v;
    try {
      k = decodeURIComponent(pair.slice(0, eq));
      v = decodeURIComponent(pair.slice(eq + 1));
    } catch {
      continue;
    }
    if (k) out[k] = v;
  }
  return out;
}

// Seeded PRNG: mulberry32 for next(), Box-Muller for normal().
export function rng(seed) {
  let a = seed >>> 0;
  let spare = null;
  function next() {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function normal() {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    do {
      u = next();
    } while (u === 0);
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * next();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  }
  return { next, normal };
}

// Round a raw step size to 1, 2 or 5 times a power of ten.
export function niceNum(x, round) {
  const exp = Math.floor(Math.log10(x));
  const f = x / 10 ** exp;
  let nf;
  if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * 10 ** exp;
}

// Tick positions between min and max, aiming for `target` ticks.
// Handles min === max and negative value ranges; min > max is swapped.
export function computeTicks(min, max, target = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { ticks: [], step: 1 };
  if (min === max) return { ticks: [min], step: 1 };
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const step = niceNum((hi - lo) / Math.max(1, target - 1), true);
  const start = Math.ceil(lo / step - 1e-9) * step;
  const n = Math.floor((hi - start) / step + 1e-9);
  const ticks = [];
  for (let i = 0; i <= n; i++) {
    const v = start + i * step;
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

// Snap a value to a step multiple if it already sits on one (within float
// noise), otherwise floor/ceil outward. Avoids padding an exact boundary.
function snapFloor(v, step) {
  const n = v / step;
  const r = Math.round(n);
  return (Math.abs(n - r) < 1e-9 ? r : Math.floor(n)) * step;
}
function snapCeil(v, step) {
  const n = v / step;
  const r = Math.round(n);
  return (Math.abs(n - r) < 1e-9 ? r : Math.ceil(n)) * step;
}

// Expand a data min/max to nice tick boundaries, padding outward by at most
// one tick step; never pads below 0 when the data min is 0.
export function dataRange(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const step = niceNum((hi - lo) / 4, true);
  let a = snapFloor(lo, step);
  let b = snapCeil(hi, step);
  if (lo === 0 && a < 0) a = 0;
  return [a, b];
}

// A small listener list. add(fn) returns an unsubscribe; emit(v) calls each
// listener in order with v. Pure, no DOM, so it can be tested in node.
export function listenerList() {
  const fns = [];
  return {
    add(fn) {
      fns.push(fn);
      return () => {
        const i = fns.indexOf(fn);
        if (i >= 0) fns.splice(i, 1);
      };
    },
    emit(v) {
      for (const fn of fns.slice()) fn(v);
    },
    size() {
      return fns.length;
    },
  };
}

// Units whose player time is a count, not a continuous quantity.
const DISCRETE_UNITS = new Set(['steps', 'samples', 'epochs', 'step']);

// Whether a player should show whole-number time. An explicit `integer` wins;
// otherwise it is on for discrete units (steps, samples, epochs, step).
export function playerInteger(unit, integer) {
  if (integer !== undefined) return !!integer;
  return DISCRETE_UNITS.has(unit);
}

// The time handed to onFrame / the readout: floored to an integer in integer
// mode, otherwise the raw value.
export function playerShownTime(t, integer) {
  return integer ? Math.floor(t) : t;
}

// Clamp model time to [0, duration]. Non-finite input becomes 0.
export function clampTime(t, duration) {
  if (!Number.isFinite(t)) return 0;
  return Math.min(duration, Math.max(0, t));
}

// Advance model time by dtMs of real time. Model time advances at
// speed * duration / playSeconds per real second. With loop, the excess
// beyond duration wraps to the start; without, it clamps at duration.
export function playerAdvance(t, dtMs, duration, playSeconds, speed, loop) {
  const dt = (dtMs / 1000) * speed * (duration / playSeconds);
  if (loop) {
    let nt = t + dt;
    while (nt >= duration) nt -= duration;
    while (nt < 0) nt += duration;
    return nt;
  }
  return clampTime(t + dt, duration);
}

// Format a tick value with decimals derived from the step. Trailing zeros are
// trimmed only after a decimal point, so 10 stays "10" (not "1").
export function fmtTick(v, step) {
  if (v === 0) return '0';
  const dec = Math.max(0, -Math.floor(Math.log10(step)) + (step < 1 ? 1 : 0));
  let s = v.toFixed(Math.min(dec, 6));
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s;
}

// Format a control readout: decimals from the step (like fmtTick), capped at
// 4 significant digits so 0.16666666666666666 shows as "0.1667".
export function formatValue(v, step) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const dec = Math.max(0, -Math.floor(Math.log10(step)) + (step < 1 ? 1 : 0));
  let s = v.toFixed(Math.min(dec, 6));
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  // cap at 4 significant digits, staying in plain notation
  if (s.replace(/[-.]/g, '').length > 4) {
    const d = Math.max(0, 3 - Math.floor(Math.log10(Math.abs(v))));
    s = v.toFixed(d);
    if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  }
  return s;
}
