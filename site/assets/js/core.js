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

// Expand [min, max] to a range whose ends sit on nice multiples of a step.
export function niceRange(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const step = niceNum((hi - lo) / 4, false);
  return [Math.floor(lo / step - 1e-9) * step, Math.ceil(hi / step + 1e-9) * step];
}
