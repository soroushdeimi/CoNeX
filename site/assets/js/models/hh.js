// Pure math for the Hodgkin-Huxley (1952) neuron. No DOM, no globals.
// Mirrors CoNeX exactly: same rates, same forward-Euler update order, same
// singularity handling. See conex/behaviors/neurons/neuron_types/hh_neurons.py.
//
// Units (squid giant axon, 6.3 degC, from Hodgkin & Huxley 1952):
//   v in mV, t in ms, I in uA/cm2, C in uF/cm2, g in mS/cm2, E in mV.
//   Gate rates alpha/beta in ms^-1.

export const HH_DEFAULTS = {
  C: 1.0,        // uF/cm2
  g_Na: 120.0,   // mS/cm2
  g_K: 36.0,     // mS/cm2
  g_L: 0.3,      // mS/cm2
  E_Na: 50.0,    // mV
  E_K: -77.0,    // mV
  E_L: -54.387,  // mV
  threshold: 0.0,// mV (spike-detection threshold, a simulation choice)
  init_v: -65.0, // mV
  dt: 0.025,     // ms
};

// |u| below this uses the Taylor series of u / (1 - exp(-u)) at u = 0.
const EPS = 1e-6;

// u / (1 - exp(-u)) with the removable singularity at u = 0 handled by the
// Taylor series 1 + u/2 + u^2/12 (matches CoNeX's torch.where branch).
function uOverOneMinusExpNegU(u) {
  if (Math.abs(u) < EPS) return 1 + 0.5 * u + (u * u) / 12;
  return -u / Math.expm1(-u);
}

// Gate rates alpha and beta in ms^-1 at voltage v (mV).
export function hhRates(v) {
  const u = (v + 40) / 10;
  const q = (v + 55) / 10;
  return {
    alpha_m: uOverOneMinusExpNegU(u),
    beta_m: 4 * Math.exp(-(v + 65) / 18),
    alpha_h: 0.07 * Math.exp(-(v + 65) / 20),
    beta_h: 1 / (1 + Math.exp(-(v + 35) / 10)),
    alpha_n: 0.1 * uOverOneMinusExpNegU(q),
    beta_n: 0.125 * Math.exp(-(v + 65) / 80),
  };
}

// Steady-state open probabilities x_inf = alpha / (alpha + beta) at voltage v.
export function hhSteadyGates(v) {
  const r = hhRates(v);
  return {
    m: r.alpha_m / (r.alpha_m + r.beta_m),
    h: r.alpha_h / (r.alpha_h + r.beta_h),
    n: r.alpha_n / (r.alpha_n + r.beta_n),
  };
}

// Initial state: v at init_v, gates at their steady state there (CoNeX does this).
export function hhInitialState(p) {
  const g = hhSteadyGates(p.init_v);
  return { v: p.init_v, m: g.m, h: g.h, n: g.n };
}

// One CoNeX iteration. Mirrors HH.forward exactly:
//   rates from the current v, Euler gate update, currents from the UPDATED gates
//   but the OLD v, Euler membrane update, spike = upward threshold crossing
//   (no reset). Returns the new state plus the currents and the spike flag.
export function hhStep(state, p, I, dt) {
  const v = state.v;
  const r = hhRates(v);
  const m = state.m + (r.alpha_m * (1 - state.m) - r.beta_m * state.m) * dt;
  const h = state.h + (r.alpha_h * (1 - state.h) - r.beta_h * state.h) * dt;
  const n = state.n + (r.alpha_n * (1 - state.n) - r.beta_n * state.n) * dt;
  const i_Na = p.g_Na * m ** 3 * h * (v - p.E_Na);
  const i_K = p.g_K * n ** 4 * (v - p.E_K);
  const i_L = p.g_L * (v - p.E_L);
  const dv = (I - i_Na - i_K - i_L) * dt / p.C;
  const spike = v < p.threshold && v + dv >= p.threshold;
  return { v: v + dv, m, h, n, i_Na, i_K, i_L, spike };
}

// Run one neuron for `steps` iterations. `I` is a number (constant) or a
// function of the step index (for a pulse). Returns the full traces.
export function hhSimulate(p, I, steps, dt, initV) {
  const iv = initV ?? p.init_v;
  let state = hhInitialState({ ...p, init_v: iv });
  const v = [], m = [], h = [], n = [], i_Na = [], i_K = [], spikes = [];
  for (let i = 0; i < steps; i++) {
    const cur = typeof I === 'function' ? I(i) : I;
    const r = hhStep(state, p, cur, dt);
    state = { v: r.v, m: r.m, h: r.h, n: r.n };
    v.push(r.v); m.push(r.m); h.push(r.h); n.push(r.n);
    i_Na.push(r.i_Na); i_K.push(r.i_K);
    spikes.push(r.spike);
  }
  return { v, m, h, n, i_Na, i_K, spikes };
}

// --- Pure mapping functions for the hero animation (AMENDMENT 5) ---

// Number of open subunits to draw for a gate with open probability `fraction`
// (0..1) and `total` subunits. Deterministic: round to the nearest integer.
export function openGates(fraction, total) {
  if (fraction < 0 || fraction > 1) {
    throw new RangeError(`gate fraction must be in [0,1], got ${fraction}`);
  }
  return Math.round(fraction * total);
}

// Particle spawn rate (particles per second of real time) from a real current
// (uA/cm2). Proportional to the magnitude of the current; `scale` sets the
// visual gain. Na+ current is inward (i_Na < 0), K+ outward (i_K > 0); the
// sign is dropped here and applied by the caller.
export function particleRate(current, scale = 1) {
  return Math.abs(current) * scale;
}

// Membrane tint fraction 0..1 from the voltage, for shading the cell interior.
export function membraneTint(v, vMin, vMax) {
  if (!(vMax > vMin)) throw new RangeError(`vMax must exceed vMin, got ${vMax} <= ${vMin}`);
  return Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin)));
}

// One-line narration driven by the model state. i_Na < 0 is inward
// (depolarizing); i_K > 0 is outward (hyperpolarizing).
export function hhNarration({ v, v_rest, i_Na, i_K, h }) {
  if (v < v_rest - 2) return 'afterhyperpolarization: v below rest, K+ still flowing out';
  if (i_K > 20) return 'K+ gates open: repolarization';
  if (i_Na < -20 && h < 0.4) return 'h closes: Na+ inactivates';
  if (i_Na < -20) return 'Na+ gates open: depolarization accelerates';
  return 'at rest';
}
