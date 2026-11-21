// Pure mapping functions for the hero animations: model state -> a drawn
// quantity (a number, a 0..1 fraction, or a narration string). No DOM, no
// ui.color, no Math.random. Each plate's canvas code calls these so the
// mapping is testable in isolation (site/tests/anim.test.mjs).

export const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const lerp = (a, b, f) => a + (b - a) * f;

// --- syn-stdp ---------------------------------------------------------------

// Halo brightness from a spike trace. A trace is ~0..1 right after a spike and
// decays toward 0, so it maps directly to a 0..1 glow.
export const stdpHalo = (trace) => clamp01(trace);

// Synapse thickness in px from the weight, mapped over [wMin, wMax].
export function stdpSynapsePx(w, wMin, wMax, minPx = 3, maxPx = 17) {
  const f = wMax > wMin ? clamp01((w - wMin) / (wMax - wMin)) : 0.5;
  return lerp(minPx, maxPx, f);
}

// delta_t (ms) to highlight on the STDP window curve for a spike at time t.
// A post spike highlights (t - lastPre)  -> pre before post (potentiation side).
// A pre spike highlights -(t - lastPost) -> post before pre (depression side).
// Returns null when there is no pairing to highlight.
export function stdpHighlightDelta(t, lastPre, lastPost, preSpike, postSpike) {
  if (postSpike && lastPre >= 0) return t - lastPre;
  if (preSpike && lastPost >= 0) return -(t - lastPost);
  return null;
}

// Progress (0..1) of a spike pulse travelling from its neuron to the synapse.
// A spike at model time t0 takes `travel` model ms to reach the synapse; the
// pulse is visible for 0 <= (t - t0) < travel. Returns null when the pulse is
// not in flight (before the spike, or after it has arrived).
export function stdpSpikePulse(t, t0, travel) {
  if (t0 < 0 || travel <= 0) return null;
  const age = t - t0;
  if (age < 0 || age >= travel) return null;
  return age / travel;
}

// One-line narration driven by the current pairing and the resulting dw.
export function stdpNarration({ preSpike, postSpike, dw }) {
  if (preSpike && postSpike) return 'pre and post fire together: potentiation and depression both apply';
  if (preSpike) return 'pre spike: pre_trace rises; it will potentiate if the post fires next';
  if (postSpike) return 'post spike: post_trace rises; it depresses if the pre fired before';
  if (dw > 1e-9) return 'pre before post: potentiation (dw > 0)';
  if (dw < -1e-9) return 'post before pre: depression (dw < 0)';
  return 'traces decay: no spike pairing this step';
}

// --- syn-istdp --------------------------------------------------------------

// Needle position (0..1) for a rate meter spanning [0, maxRate].
export const istdpMeterPos = (rate, maxRate) => (maxRate > 0 ? clamp01(rate / maxRate) : 0);

// Inhibitory synapse thickness in px from the weight over [0, wMax].
export function istdpSynapsePx(w, wMax, minPx = 2, maxPx = 14) {
  const f = wMax > 0 ? clamp01(w / wMax) : 0;
  return lerp(minPx, maxPx, f);
}

// Narration: above target the inhibitory weight grows, below target it shrinks.
export function istdpNarration(rate, rho) {
  if (rate > rho * 1.05) return 'post rate above target: inhibitory weight grows to suppress it';
  if (rate < rho * 0.95) return 'post rate below target: inhibitory weight shrinks';
  return 'post rate at target rho: the weight is stable';
}

// --- syn-rstdp --------------------------------------------------------------

// Eligibility glow (0..1) from the eligibility trace c, scaled by its max |c|.
export const rstdpEligGlow = (c, cMax) => (cMax > 0 ? clamp01(Math.abs(c) / cMax) : 0);

// Dopamine cloud glow (0..1) from the dopamine concentration.
export const rstdpDopGlow = (dop, dopMax) => (dopMax > 0 ? clamp01(dop / dopMax) : 0);

// Narration driven by whether the post fired inside the reward window.
export function rstdpNarration({ postSpike, inWindow, dop }) {
  if (postSpike && inWindow) return 'post fires in the reward window: dopamine released, eligibility becomes weight change';
  if (postSpike && !inWindow) return 'post fires outside the window: no reward, eligibility decays';
  if (dop > 1e-6) return 'dopamine present: weight += eligibility * dopamine';
  return 'eligibility trace decays (tau_c); waiting for a reward';
}

// --- syn-connections --------------------------------------------------------

// Progress (0..1) of a spike particle travelling along its connection.
export const connProgress = (t) => clamp01(t);

// Post-neuron current bar fill fraction (0..1, magnitude only).
export const connBarFill = (current, yMax) => (yMax > 0 ? clamp01(Math.abs(current) / yMax) : 0);

// --- enc-poisson ------------------------------------------------------------

// Progress (0..1) of a spark flying from its pixel into the raster.
export const encSparkProgress = (t) => clamp01(t);

// Narration: which code the selected encoder implements.
export function encNarration(encoder) {
  if (encoder === 'i2l') return 'latency code: brighter pixels spike sooner, the first spike carries the value';
  if (encoder === 'poisson') return 'rate code: inter-spike intervals ~ Poisson(1 / (intensity * ratio))';
  return 'rate code: each pixel spikes with probability intensity * ratio per step';
}

// --- learn-stdp-family ------------------------------------------------------

// Cell size/colour fraction (0..1) for a weight in [wMin, wMax].
export function stdpFamCellFrac(w, wMin, wMax) {
  return wMax > wMin ? clamp01((w - wMin) / (wMax - wMin)) : 0.5;
}

// Narration: what the selected rule/topology is doing.
export function stdpFamNarration(rule, connection) {
  const top = connection === 'Conv2d' ? 'a shared kernel swept across the input'
    : connection === 'Local2d' ? 'a local kernel per output position'
    : connection === 'One2One' ? 'one-to-one diagonal synapses'
    : connection === 'Sparse' ? 'only the existing sparse connections'
    : 'the full dense weight matrix';
  if (rule === 'iSTDP') return `iSTDP tunes ${top} toward the target rate rho`;
  if (rule === 'RSTDP') return `RSTDP stores eligibility on ${top}; dopamine gates the weight change`;
  return `STDP updates ${top} from the pre/post spike traces`;
}

// --- learn-bounds -----------------------------------------------------------

// Bead position (0..1) along the rail from wMin to wMax.
export function boundsBeadPos(w, wMin, wMax) {
  return wMax > wMin ? clamp01((w - wMin) / (wMax - wMin)) : 0.5;
}

// Spring-stiffness visual (0..1) from the bound factor B(w): a stiff spring
// (high factor) means plasticity is still possible.
export const boundsSpring = (factor) => clamp01(factor);

// Narration for one bound regime at the current weight and phase.
export function boundsNarration(bound, w, wMin, wMax, phase) {
  const atMax = w >= wMax - 1e-9;
  const atMin = w <= wMin + 1e-9;
  if (bound === 'hard_bound' && (atMax || atMin)) return 'hard_bound: at the boundary the factor is 0, the weight is stuck';
  if (bound === 'soft_bound' && (atMax || atMin)) return 'soft_bound: at the boundary the factor is 0, updates stop smoothly';
  if (phase === 'plus') return `${bound}: potentiation pushes the weight up`;
  return `${bound}: depression pushes the weight down`;
}
