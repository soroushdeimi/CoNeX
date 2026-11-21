// Pure math for the Neurons area. No DOM, no globals.
// Mirrors CoNeX exactly: same equations, same parameter names, same update order.
//
// Units (physiological, from Gerstner et al. "Neuronal Dynamics" 2014, Table 6.1):
//   v in mV, t in ms, I in pA, R in GOhm (so R*I is in mV), a in nS (so a*(v-v_rest)
//   is in pA), b in pA, tau/tau_w in ms.
//
// CoNeX has no built-in numeric defaults for the neuron parameters (they are all
// required constructor args). DEFAULTS is the physiological set used by the plates;
// AU_DEFAULTS is a dimensionless (a.u.) set kept for the original golden cases.

export const DEFAULTS = {
  R: 0.5,          // GOhm  (500 MOhm)
  threshold: 0.0,  // mV (AdEx spike-detection threshold; a simulation choice, not from Table 6.1)
  tau: 20.0,       // ms
  v_reset: -55.0,  // mV
  v_rest: -70.0,   // mV
  // ELIF
  delta: 2.0,      // mV
  theta_rh: -50.0, // mV
  // AELIF
  alpha: 0.0,      // nS
  beta: 60.0,      // pA
  w_tau: 30.0,     // ms
  dt: 0.1,         // ms
};

export const AU_DEFAULTS = {
  R: 10.0,
  threshold: 20.0,
  tau: 10.0,
  v_reset: 0.0,
  v_rest: 0.0,
  delta: 2.0,
  theta_rh: 15.0,
  alpha: 0.08,
  beta: 0.5,
  w_tau: 100.0,
  dt: 1.0,
};

// Gerstner et al. Table 6.1 AdEx firing patterns. Shared for every row:
// R=0.5 GOhm, threshold=0 mV, v_rest=-70 mV, delta=2 mV, theta_rh=-50 mV, dt=0.1 ms.
// CoNeX names: alpha=a, beta=b, w_tau=tau_w, v_reset=u_r.
export const TABLE61 = {
  'Tonic':         { tau: 20.0, alpha: 0.0,  w_tau: 30.0,  beta: 60.0, v_reset: -55.0, I: 65.0 },
  'Adapting':      { tau: 20.0, alpha: 0.0,  w_tau: 100.0, beta: 5.0,  v_reset: -55.0, I: 65.0 },
  'Initial burst': { tau: 5.0,  alpha: 0.5,  w_tau: 100.0, beta: 7.0,  v_reset: -51.0, I: 65.0 },
  'Bursting':      { tau: 5.0,  alpha: -0.5, w_tau: 100.0, beta: 7.0,  v_reset: -46.0, I: 65.0 },
  'Irregular':     { tau: 9.9,  alpha: -0.5, w_tau: 100.0, beta: 7.0,  v_reset: -46.0, I: 65.0 },
  'Transient':     { tau: 10.0, alpha: 1.0,  w_tau: 100.0, beta: 10.0, v_reset: -60.0, I: 65.0 },
  'Delayed':       { tau: 5.0,  alpha: -1.0, w_tau: 100.0, beta: 10.0, v_reset: -60.0, I: 25.0 },
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

// Spike times (ms) from a simulate() result, given dt.
export function spikeTimes(spikes, dt) {
  const t = [];
  for (let i = 0; i < spikes.length; i++) if (spikes[i]) t.push(i * dt);
  return t;
}

// Measurable firing-pattern statistics from spike times (ms).
// cv: coefficient of variation of ISIs. early/late: mean ISI of the first/last
// quarter of ISIs. Used to check whether a Table 6.1 pattern reproduces.
export function firingStats(t) {
  const isi = [];
  for (let i = 1; i < t.length; i++) isi.push(t[i] - t[i - 1]);
  const n = isi.length;
  if (n < 2) return { n: t.length, first_t: t[0] ?? null, last_t: t[t.length - 1] ?? null, cv: null, early: null, late: null };
  const mean = isi.reduce((a, b) => a + b, 0) / n;
  const cv = Math.sqrt(isi.reduce((a, b) => a + (b - mean) ** 2, 0) / n) / mean;
  const q = Math.max(1, Math.floor(n / 4));
  const early = isi.slice(0, q).reduce((a, b) => a + b, 0) / q;
  const late = isi.slice(-q).reduce((a, b) => a + b, 0) / q;
  return { n: t.length, first_t: t[0], last_t: t[t.length - 1], cv, early, late };
}

// Smallest constant current that makes a LIF fire: I_rheo = (threshold - v_rest)/R.
export function lifRheobase(p) {
  assertPositive('R', p.R);
  return (p.threshold - p.v_rest) / p.R;
}

// Leaky-bucket picture of the LIF. The "water level" is v; two flow rates
// (mV/ms) move it. Inflow from the tap is the input term R*I/tau; outflow
// through the hole is the leak (v - v_rest)/tau. Together they are exactly the
// LIF update dv/dt = (R*I + (v_rest - v)) / tau. `params` needs R, I, tau, v_rest.
export function bucketFlows(params, v) {
  assertPositive('tau', params.tau);
  const input = (params.R * params.I) / params.tau;
  const leak = (v - params.v_rest) / params.tau;
  return { input, leak };
}

// Leaky-bucket water level as a fraction 0..1 between v_rest (empty) and
// threshold (rim), clamped. This is the real v at the player time — the level
// must track v(t) at all times (it drops to v_reset at the spike step, which is
// the real reset). `params` needs v_rest and threshold.
export function bucketLevel(v, params) {
  const span = params.threshold - params.v_rest;
  if (!(span > 0)) throw new RangeError(`threshold must exceed v_rest, got ${params.threshold} <= ${params.v_rest}`);
  return Math.max(0, Math.min(1, (v - params.v_rest) / span));
}

// Splash window: true while `t` is within `windowMs` of model time after a spike
// at `lastSpikeT`. A spike at time 0 is active for t in [0, windowMs). No spike
// (lastSpikeT null) is never active.
export function splashActive(t, lastSpikeT, windowMs = 0.5) {
  if (lastSpikeT == null) return false;
  return t >= lastSpikeT && t < lastSpikeT + windowMs;
}

// LIF hero narration: which of the three regimes the neuron is in, as a one-line
// caption. While splashing (just spiked) it reports the reset; otherwise it
// compares the two real flows from bucketFlows — if the input flow exceeds the
// leak flow the voltage is rising, else it is settling back toward steady state.
export function lifNarration({ v, p, I, splashing }) {
  if (splashing) return 'threshold crossed: spike, reset to v_reset';
  const { input, leak } = bucketFlows({ R: p.R, I, tau: p.tau, v_rest: p.v_rest }, v);
  return input > leak ? 'input beats leak: v rises' : 'leak balances input: v settles below threshold';
}

// Decorative droplets for the bucket hero. The expected count is proportional to
// the REAL flow (mV/ms) times the frame time; spawn the whole part plus one more
// with probability equal to the fractional part (seeded rng). Capped. Pure.
export function dropletCount(flow, frameSeconds, rng, rate = 30, cap = 40) {
  if (!Number.isFinite(flow) || flow <= 0) return 0;
  const expected = flow * frameSeconds * rate;
  const whole = Math.floor(expected);
  const extra = rng.next() < (expected - whole) ? 1 : 0;
  return Math.min(cap, whole + extra);
}

// ELIF exponential spike-initiation term (mV) at voltage v: delta*exp((v-theta_rh)/delta).
// This is the "pump" in the bucket picture: it is the extra inflow that grows
// sharply as v approaches theta_rh and drives the take-off.
export function elifExpTerm(v, p) {
  assertPositive('delta', p.delta);
  return p.delta * Math.exp((v - p.theta_rh) / p.delta);
}

// ELIF hero narration, driven by the real quantities. While splashing it reports
// the reset; otherwise it compares the exponential inflow (delta*exp(...)/tau,
// mV/ms) to the input flow (R*I/tau, mV/ms) — the take-off begins when the
// exponential term rivals the input.
export function elifNarration({ v, p, I, splashing }) {
  if (splashing) return 'threshold crossed: spike, reset to v_reset';
  const pump = elifExpTerm(v, p) / p.tau;
  const input = (p.R * I) / p.tau;
  return pump > input
    ? 'exponential term takes over: v accelerates to threshold'
    : 'v below take-off: input and leak balance';
}

// AELIF hero narration, driven by the real quantities. While splashing it reports
// the reset; otherwise it compares the adaptation current (R*w) to the input
// (R*I): the net drive is R*(I - w), so when w exceeds I the adaptation pulls v
// down and firing slows.
export function aelifNarration({ v, omega, p, I, splashing }) {
  if (splashing) return 'threshold crossed: spike, reset to v_reset';
  return omega > I
    ? 'adaptation w exceeds input: v is pulled down, firing slows'
    : 'input beats adaptation: v rises toward threshold';
}

// Homeostasis hero narration, driven by the real group rate vs the target. For
// activity-based control the threshold moves to bring the rate to the target:
// above target the threshold rises (slows firing), below it the threshold falls.
// For voltage-based control the voltage is pushed toward the target band.
export function homeostasisNarration({ rate, target, mode }) {
  if (mode === 'voltage') return 'voltage pushed toward the target band';
  if (rate > target) return 'rate above target: threshold rises to slow firing';
  if (rate < target) return 'rate below target: threshold falls to speed firing';
  return 'rate at target: threshold holds';
}

// Analytic LIF firing rate (Hz, in a.u. of time): f = 1 / (tau * ln((vss-v_reset)/(vss-threshold)))
// where vss = v_rest + R*I. The formula is only valid in the normal firing regime
// v_reset < threshold < vss; outside it (e.g. v_reset >= threshold, or I at/below the
// rheobase) it returns 0 rather than NaN/Infinity.
export function lifRateAnalytic(p, I) {
  const vss = p.v_rest + p.R * I;
  if (vss <= p.threshold) return 0.0;
  const arg = (vss - p.v_reset) / (vss - p.threshold);
  if (!(arg > 0)) return 0.0;
  const T = p.tau * Math.log(arg);
  if (!(T > 0)) return 0.0;
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
