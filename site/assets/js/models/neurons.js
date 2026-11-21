// Pure math for the Neurons area. No DOM, no globals.
// Mirrors CoNeX exactly: same equations, same parameter names, same update order.
// All quantities are in CoNeX dimensionless units (a.u.) unless a unit is stated.
//
// CoNeX has no built-in numeric defaults for the neuron parameters (they are all
// required constructor args), so this module defines one canonical set used as the
// site default and as the golden reference:
//   R=10, threshold=20, tau=10, v_reset=0, v_rest=0 (a.u.)

export const DEFAULTS = {
  R: 10.0,
  threshold: 20.0,
  tau: 10.0,
  v_reset: 0.0,
  v_rest: 0.0,
  // ELIF
  delta: 2.0,
  theta_rh: 15.0,
  // AELIF
  alpha: 0.08,
  beta: 0.5,
  w_tau: 100.0,
  dt: 1.0,
};

function assertPositive(name, value) {
  if (typeof value !== 'number' || !isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive finite number, got ${value}`);
  }
}

// --- Single-step dynamics (one CoNeX iteration for one neuron) ---
// Each returns {v, spike} (AELIF also returns omega). The spike is decided on the
// post-update voltage, exactly as CoNeX's Fire does (v >= threshold, then reset).

// LIF.forward only: the voltage update, without the Fire step. Used when another
// behavior (e.g. KWTA) must run between the update and the spike decision.
export function lifUpdate(state, p, I, dt) {
  assertPositive('tau', p.tau);
  const Fu = p.v_rest - state.v;
  const RIu = p.R * I;
  return state.v + (Fu + RIu) * dt / p.tau;
}

export function lifStep(state, p, I, dt) {
  const v = lifUpdate(state, p, I, dt);
  const spike = v >= p.threshold;
  return { v: spike ? p.v_reset : v, spike };
}

export function elifStep(state, p, I, dt) {
  assertPositive('tau', p.tau);
  assertPositive('delta', p.delta);
  const Fu = p.v_rest - state.v + p.delta * Math.exp((state.v - p.theta_rh) / p.delta);
  const RIu = p.R * I;
  let v = state.v + (Fu + RIu) * dt / p.tau;
  const spike = v >= p.threshold;
  if (spike) v = p.v_reset;
  return { v, spike };
}

export function aelifStep(state, p, I, dt) {
  assertPositive('tau', p.tau);
  assertPositive('delta', p.delta);
  assertPositive('w_tau', p.w_tau);
  const Fu = p.v_rest - state.v + p.delta * Math.exp((state.v - p.theta_rh) / p.delta);
  const RIu = p.R * I - p.R * state.omega;
  let v = state.v + (Fu + RIu) * dt / p.tau;
  // Fire: decide spike and reset v, THEN update omega using the post-reset v.
  const spike = v >= p.threshold;
  if (spike) v = p.v_reset;
  const spikeAdapt = p.beta * p.w_tau * (spike ? 1 : 0);
  const subThresh = p.alpha * (v - p.v_rest);
  const domega = (subThresh - state.omega + spikeAdapt) * dt / p.w_tau;
  return { v, omega: state.omega + domega, spike };
}

const STEPS = { lif: lifStep, elif: elifStep, aelif: aelifStep };

// Run one neuron for `steps` iterations with a constant current I.
// Returns {v:[], spikes:[], omega:[] (aelif only)}.
export function simulate(model, p, I, steps, dt = 1.0, initV = 0.0) {
  const step = STEPS[model];
  if (!step) throw new RangeError(`unknown model "${model}"`);
  let state = { v: initV, omega: 0.0 };
  const v = [], spikes = [], omega = [];
  for (let i = 0; i < steps; i++) {
    const r = step(state, p, I, dt);
    state = { v: r.v, omega: r.omega ?? state.omega };
    v.push(r.v);
    spikes.push(r.spike);
    if (model === 'aelif') omega.push(r.omega);
  }
  return { v, spikes, omega };
}

// Smallest constant current that makes a LIF fire: I_rheo = (threshold - v_rest)/R.
export function lifRheobase(p) {
  assertPositive('R', p.R);
  return (p.threshold - p.v_rest) / p.R;
}

// Analytic LIF firing rate (Hz, in a.u. of time): f = 1 / (tau * ln((vss-v_reset)/(vss-threshold)))
// where vss = v_rest + R*I. Returns 0 when I is at or below the rheobase.
export function lifRateAnalytic(p, I) {
  const vss = p.v_rest + p.R * I;
  if (vss <= p.threshold) return 0.0;
  const T = p.tau * Math.log((vss - p.v_reset) / (vss - p.threshold));
  return 1.0 / T;
}

// f-I curve: spike count per current, simulating each current for `steps` steps.
export function fiCurve(model, p, currents, steps, dt = 1.0) {
  return currents.map((c) => simulate(model, p, c, steps, dt).spikes.reduce((a, b) => a + (b ? 1 : 0), 0));
}

// KWTA as a pure function. Given a voltage vector, the k winners (top-k by v)
// keep their voltage; every other neuron above threshold is reset to v_reset.
// If at most k neurons are above threshold, nothing changes. Mirrors CoNeX KWTA.forward.
export function kwta(v, threshold, v_reset, k) {
  if (!Number.isInteger(k) || k < 0) throw new RangeError(`k must be a non-negative integer, got ${k}`);
  const n = v.length;
  const willSpike = v.map((x) => x >= threshold);
  if (willSpike.reduce((a, b) => a + (b ? 1 : 0), 0) <= k) return v.slice();
  const idx = v.map((x, i) => i).sort((a, b) => v[b] - v[a]);
  const winners = new Set(idx.slice(0, k));
  return v.map((x, i) => (willSpike[i] && !winners.has(i) ? v_reset : x));
}

// InherentNoise: v += uniform[0,1) * scale + offset, using a seeded rng.
export function inherentNoise(v, rng, scale, offset) {
  return v.map((x) => x + rng.next() * scale + offset);
}

// ActivityBaseHomeostasis (per-neuron, mirrors CoNeX). CoNeX keeps `activities`
// and `threshold` as per-neuron vectors, so each neuron runs its own copy: +1 per
// spike, -activity_rate/(window_size - activity_rate) per non-spike. Every
// window_size steps the neuron's threshold shifts by -activities*updating_rate
// and updating_rate decays by decay_rate.
export function makeActivityHomeostasis({ activity_rate, window_size, updating_rate, decay_rate = 1.0 }) {
  if (window_size <= activity_rate) {
    throw new RangeError(`window_size must exceed activity_rate, got ${window_size} <= ${activity_rate}`);
  }
  const firingReward = 1;
  const nonFiringPenalty = -activity_rate / (window_size - activity_rate);
  return {
    activities: 0,
    updating_rate,
    // spike: boolean for this neuron this step; threshold: number; iteration: 1-based.
    step(spike, threshold, iteration) {
      this.activities += spike ? firingReward : nonFiringPenalty;
      if (iteration % window_size === 0) {
        const change = -this.activities * this.updating_rate;
        const newThreshold = threshold - change;
        this.activities = 0;
        this.updating_rate *= decay_rate;
        return newThreshold;
      }
      return threshold;
    },
  };
}

// VoltageBaseHomeostasis. Per step: push v toward the [min_ta, max_ta] band by
// accumulating an "exhaustion" term. Mirrors CoNeX VoltageBaseHomeostasis.forward.
export function makeVoltageHomeostasis({ target_voltage, max_ta, min_ta, eta_ip = 0.001 }) {
  const max = max_ta ?? target_voltage;
  const min = min_ta ?? target_voltage;
  return {
    exhaustion: 0,
    step(v) {
      const greater = v > max ? v - max : 0;
      const smaller = v < min ? v - min : 0;
      this.exhaustion += (greater + smaller) * eta_ip;
      return v - this.exhaustion;
    },
  };
}
