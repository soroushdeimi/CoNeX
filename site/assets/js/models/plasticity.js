// Pure math for the plasticity & encoding area. Mirrors CoNeX Python:
// conex/behaviors/synapses/{learning,specs,dendrites}.py,
// conex/behaviors/network/{neuromodulators,payoff,time_resolution}.py,
// conex/behaviors/neurons/{lif_neurons,axon}.py,
// conex/helpers/transforms/encoders.py.
// No DOM. All time in ms, rates in Hz, weights dimensionless (a.u.).

// mulberry32, same sequence as ui.rng(seed).next()
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function checkFinite(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number, got ${String(value)}`);
  }
}

// --- encoders (conex/helpers/transforms/encoders.py) ---

// img: flat array of values in [0,1]. Returns flat bool array
// [time_window * n], row-major (time first), like torch (T, ...shape).
export function simplePoisson(img, { time_window, ratio, seed = 0 } = {}) {
  checkFinite('time_window', time_window);
  checkFinite('ratio', ratio);
  if (time_window < 1 || !Number.isInteger(time_window)) {
    throw new Error(`time_window must be a positive integer, got ${time_window}`);
  }
  const rand = mulberry32(seed);
  const n = img.length;
  const out = new Array(time_window * n).fill(false);
  for (let t = 0; t < time_window; t++) {
    for (let i = 0; i < n; i++) {
      const p = img[i] * ratio;
      out[t * n + i] = p >= rand();
    }
  }
  return out;
}

// Knuth Poisson sample; consumes rand() in the same order as torch.distributions.Poisson
function poissonSample(lambda, rand) {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= rand();
  } while (p > L);
  return k - 1;
}

export function poisson(img, { time_window, ratio, seed = 0 } = {}) {
  checkFinite('time_window', time_window);
  checkFinite('ratio', ratio);
  if (time_window < 1 || !Number.isInteger(time_window)) {
    throw new Error(`time_window must be a positive integer, got ${time_window}`);
  }
  const rand = mulberry32(seed);
  const n = img.length;
  const out = new Array(time_window * n).fill(false);
  for (let i = 0; i < n; i++) {
    const rate = img[i] * ratio;
    const lambda = rate === 0 ? 0 : 1 / rate;
    // torch: intervals ~ Poisson(1/rate); zero intervals bumped to 1 for nonzero pixels
    const intervals = [];
    for (let s = 0; s < time_window; s++) {
      let iv = poissonSample(lambda, rand);
      if (rate !== 0 && iv === 0) iv = 1;
      intervals.push(iv);
    }
    // torch: times = cumsum(intervals); times[times >= time_window+1] = 0; spikes[times]=True; spikes[1:]
    // a spike at cumsum value c lands in output row c-1 (after spikes[1:]).
    // cumsum is non-decreasing, so once it overflows it stays overflowed -> break
    let t = 0;
    for (let s = 0; s < time_window; s++) {
      t += intervals[s];
      if (t >= time_window + 1) break;
      if (t > 0) out[(t - 1) * n + i] = true;
    }
  }
  return out;
}

// quantile with linear interpolation, matching torch.quantile default
export function quantile(values, q) {
  const xs = [...values].sort((a, b) => a - b);
  const pos = q * (xs.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo);
}

export function intensity2Latency(img, { time_window, threshold = null, sparsity = null, min_val = 0.0, max_val = 1.0, lower_trim = true, higher_trim = true } = {}) {
  checkFinite('time_window', time_window);
  if (time_window < 1 || !Number.isInteger(time_window)) {
    throw new Error(`time_window must be a positive integer, got ${time_window}`);
  }
  const n = img.length;
  let thr = threshold === null ? min_val : threshold;
  if (sparsity !== null) thr = quantile(img, 1 - sparsity);
  const below = img.map((x) => x < thr);
  let work = img.map((x) => x - min_val);
  const max_value = max_val - min_val;
  let max_factor = 1 / max_value;
  if (lower_trim && !below.every(Boolean)) {
    const img_min = Math.min(...work.filter((_, i) => !below[i]));
    work = work.map((x) => x - img_min);
    max_factor = 1 / (max_value - img_min);
  }
  if (higher_trim && Math.max(...work) !== 0) {
    max_factor = 1 / Math.max(...work);
  }
  const index = work.map((x) => Math.max(0, Math.ceil(x * max_factor * (time_window - 1)) + 1));
  for (let i = 0; i < index.length; i++) if (below[i]) index[i] = 0;
  // torch: spikes (time_window+1, n); scatter_(0, index, True); spikes = spikes[1:]; return flip(0)
  const spikes = new Array((time_window + 1) * n).fill(false);
  for (let i = 0; i < n; i++) if (index[i] > 0) spikes[index[i] * n + i] = true;
  // out[t] = spikes[time_window - t] after [1:] then flip(0)
  const out = new Array(time_window * n).fill(false);
  for (let t = 0; t < time_window; t++) {
    for (let i = 0; i < n; i++) out[t * n + i] = spikes[(time_window - t) * n + i];
  }
  return out;
}

// --- spike traces (conex/behaviors/synapses/specs.py PreTrace/PostTrace) ---
// trace += spike * spike_scale; trace -= (trace / tau_s) * dt
// Both lines are in-place, so the decay acts on the already-updated trace.
export function stepTrace(trace, spike, { tau_s, spike_scale = 1.0, dt } = {}) {
  checkFinite('tau_s', tau_s);
  checkFinite('dt', dt);
  if (tau_s <= 0) throw new Error(`tau_s must be > 0, got ${tau_s}`);
  const updated = trace + spike * spike_scale;
  return updated - (updated / tau_s) * dt;
}

// --- weight bounds (conex/behaviors/synapses/learning.py) ---
export const softBound = (w, w_min, w_max) => (w - w_min) * (w_max - w);
export const hardBound = (w, w_min, w_max) => (w > w_min && w < w_max ? 1 : 0);
export const noBound = () => 1;
export const BOUNDS = { soft_bound: softBound, hard_bound: hardBound, no_bound: noBound };

function boundFn(name) {
  if (typeof name === 'function') return name;
  if (name === null || name === undefined) return noBound;
  if (!BOUNDS[name]) throw new Error(`unknown bound "${name}", expected one of ${Object.keys(BOUNDS).join(', ')}`);
  return BOUNDS[name];
}

// --- SimpleSTDP (conex/behaviors/synapses/learning.py) ---
// dw = outer(pre_trace, post_spike) * a_plus * p_bound - outer(pre_spike, post_trace) * a_minus * n_bound
// weights shape (nSrc, nDst), row-major. pre_* length nSrc, post_* length nDst.
export function stdpDw({ weights, pre_spike, post_spike, pre_trace, post_trace, a_plus, a_minus, w_min = 0.0, w_max = 1.0, positive_bound = 'no_bound', negative_bound = 'no_bound' }) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  const nSrc = pre_spike.length;
  const nDst = post_spike.length;
  if (weights.length !== nSrc * nDst) {
    throw new Error(`weights length ${weights.length} != nSrc*nDst ${nSrc * nDst}`);
  }
  const p = boundFn(positive_bound);
  const n = boundFn(negative_bound);
  const dw = new Array(nSrc * nDst).fill(0);
  for (let s = 0; s < nSrc; s++) {
    for (let d = 0; d < nDst; d++) {
      const w = weights[s * nDst + d];
      dw[s * nDst + d] =
        pre_trace[s] * post_spike[d] * a_plus * p(w, w_min, w_max) -
        pre_spike[s] * post_trace[d] * a_minus * n(w, w_min, w_max);
    }
  }
  return dw;
}

// --- SimpleiSTDP (conex/behaviors/synapses/learning.py) ---
// alpha = 2 * rho * tau_s / 1000 when alpha is None (tau_s in ms)
export function istdpAlpha({ rho, alpha = null, tau_s }) {
  if (alpha !== null && alpha !== undefined) return alpha;
  checkFinite('rho', rho);
  checkFinite('tau_s', tau_s);
  return (2 * rho * tau_s) / 1000;
}

export function istdpDw({ weights, pre_spike, post_spike, pre_trace, post_trace, rho, alpha = null, lr = 1e-5, is_inhibitory = true, tau_s }) {
  checkFinite('lr', lr);
  const a = istdpAlpha({ rho, alpha, tau_s });
  const change_sign = 1 - (is_inhibitory ? 1 : 0) * 2;
  const nSrc = pre_spike.length;
  const nDst = post_spike.length;
  if (weights.length !== nSrc * nDst) {
    throw new Error(`weights length ${weights.length} != nSrc*nDst ${nSrc * nDst}`);
  }
  const dw = new Array(nSrc * nDst).fill(0);
  for (let s = 0; s < nSrc; s++) {
    for (let d = 0; d < nDst; d++) {
      const pre_spike_changes = pre_spike[s] * (a - post_trace[d]) * change_sign;
      const post_spike_changes = pre_trace[s] * post_spike[d];
      dw[s * nDst + d] = lr * (pre_spike_changes + post_spike_changes);
    }
  }
  return dw;
}

// --- SimpleRSTDP (conex/behaviors/synapses/learning.py) ---
// c += -c/tau_c + stdpDw;  weights += c * dopamine_concentration
export function rstdpStep({ c, dw, tau_c, dopamine_concentration }) {
  checkFinite('tau_c', tau_c);
  checkFinite('dopamine_concentration', dopamine_concentration);
  if (tau_c <= 0) throw new Error(`tau_c must be > 0, got ${tau_c}`);
  const cNew = c.map((ci, i) => ci - ci / tau_c + dw[i]);
  const dwApplied = cNew.map((ci) => ci * dopamine_concentration);
  return { c: cNew, dwApplied };
}

// --- Dopamine (conex/behaviors/network/neuromodulators.py) ---
// d += -(d / tau_dopamine) + payoff
export function dopamineStep({ dopamine_concentration, tau_dopamine, payoff }) {
  checkFinite('tau_dopamine', tau_dopamine);
  checkFinite('payoff', payoff);
  if (tau_dopamine <= 0) throw new Error(`tau_dopamine must be > 0, got ${tau_dopamine}`);
  const dd_dt = -(dopamine_concentration / tau_dopamine) + payoff;
  return dopamine_concentration + dd_dt;
}

// --- LIF (conex/behaviors/neurons/neuron_types/lif_neurons.py) ---
// v += ((v_rest - v) + R * I) * dt / tau;  spike if v >= threshold, then v = v_reset
export function lifStep({ v, I, R, threshold, tau, v_reset, v_rest, dt }) {
  checkFinite('tau', tau);
  checkFinite('dt', dt);
  if (tau <= 0) throw new Error(`tau must be > 0, got ${tau}`);
  const vNew = v + ((v_rest - v) + R * I) * (dt / tau);
  const spike = vNew >= threshold;
  return { v: spike ? v_reset : vNew, spike };
}

// --- NeuronAxon (conex/behaviors/neurons/axon.py) ---
// spike_history rolls down; get_spike reads row `delay` (0 = current step)
export function axonStep({ history, spikes, max_delay }) {
  const out = new Array(max_delay).fill(0);
  for (let i = 0; i < max_delay - 1; i++) out[i] = history[i];
  out[max_delay - 1] = spikes;
  return out;
}
export function axonGetSpike(history, delay) {
  if (delay < 0 || delay >= history.length) {
    throw new Error(`delay ${delay} out of range [0, ${history.length})`);
  }
  return history[delay];
}

// --- dendritic inputs (conex/behaviors/synapses/dendrites.py) ---
// current_type = -1 if src is inhibitory, else 1. I = current_coef * current_type * calculate_input.
function currentType(inhibitory) {
  return inhibitory ? -1 : 1;
}

// SimpleDendriticInput: I[dst] = sum over src of weights[src][dst] * pre_spike[src]
// weights shape (nSrc, nDst) row-major.
export function simpleDendriticInput({ pre_spike, weights, nDst, current_coef = 1, inhibitory = false }) {
  const nSrc = pre_spike.length;
  if (weights.length !== nSrc * nDst) {
    throw new Error(`weights length ${weights.length} != nSrc*nDst ${nSrc * nDst}`);
  }
  const I = new Array(nDst).fill(0);
  for (let s = 0; s < nSrc; s++) {
    if (!pre_spike[s]) continue;
    for (let d = 0; d < nDst; d++) I[d] += weights[s * nDst + d];
  }
  const coef = current_coef * currentType(inhibitory);
  return I.map((x) => coef * x);
}

// One2OneDendriticInput: I[i] = pre_spike[i] * weights[i]; weights length nSrc == nDst
export function one2oneDendriticInput({ pre_spike, weights, current_coef = 1, inhibitory = false }) {
  if (weights.length !== pre_spike.length) {
    throw new Error(`weights length ${weights.length} != nSrc ${pre_spike.length}`);
  }
  const coef = current_coef * currentType(inhibitory);
  return pre_spike.map((s, i) => coef * s * weights[i]);
}

// SparseDendriticInput: I = pre_spike @ weights (weights (nSrc, nDst)); same math as simple
export function sparseDendriticInput({ pre_spike, weights, nDst, current_coef = 1, inhibitory = false }) {
  return simpleDendriticInput({ pre_spike, weights, nDst, current_coef, inhibitory });
}

// AveragePool2D: adaptive_avg_pool2d of pre_spike (depth, height, width) to (dstH, dstW)
// For depth==1 this is a 2D average pool. Returns flat (dstH*dstW).
export function averagePool2D({ pre_spike, srcH, srcW, dstH, dstW, current_coef = 1, inhibitory = false }) {
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW ${srcH * srcW}`);
  }
  const out = new Array(dstH * dstW).fill(0);
  for (let oh = 0; oh < dstH; oh++) {
    for (let ow = 0; ow < dstW; ow++) {
      const h0 = Math.floor((oh * srcH) / dstH);
      const h1 = Math.ceil(((oh + 1) * srcH) / dstH);
      const w0 = Math.floor((ow * srcW) / dstW);
      const w1 = Math.ceil(((ow + 1) * srcW) / dstW);
      let sum = 0;
      let cnt = 0;
      for (let h = h0; h < h1; h++) {
        for (let w = w0; w < w1; w++) {
          sum += pre_spike[h * srcW + w];
          cnt += 1;
        }
      }
      out[oh * dstW + ow] = sum / cnt;
    }
  }
  const coef = current_coef * currentType(inhibitory);
  return out.map((x) => coef * x);
}

// Conv2dDendriticInput: I = F.conv2d(pre_spike.view(1,1,H,W), weights(1,1,kh,kw), stride, padding).view(-1)
// PyTorch conv2d is cross-correlation. weights shape (1,1,kh,kw) flattened row-major.
export function conv2dDendriticInput({ pre_spike, srcH, srcW, weights, kh, kw, stride = 1, padding = 0, current_coef = 1, inhibitory = false }) {
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW ${srcH * srcW}`);
  }
  if (weights.length !== kh * kw) {
    throw new Error(`weights length ${weights.length} != kh*kw ${kh * kw}`);
  }
  const outH = Math.floor((srcH + 2 * padding - kh) / stride) + 1;
  const outW = Math.floor((srcW + 2 * padding - kw) / stride) + 1;
  const out = new Array(outH * outW).fill(0);
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      let sum = 0;
      for (let a = 0; a < kh; a++) {
        for (let b = 0; b < kw; b++) {
          const ih = oh * stride + a - padding;
          const iw = ow * stride + b - padding;
          const val = (ih < 0 || ih >= srcH || iw < 0 || iw >= srcW) ? 0 : pre_spike[ih * srcW + iw];
          sum += val * weights[a * kw + b];
        }
      }
      out[oh * outW + ow] = sum;
    }
  }
  const coef = current_coef * currentType(inhibitory);
  return out.map((x) => coef * x);
}

// Local2dDendriticInput: each dst position j has its own kernel over its own patch.
// I[j] = sum_k patch(j,k) * weights[j*conn + k]; weights shape (1, out_size, conn_size) flattened.
export function local2dDendriticInput({ pre_spike, srcH, srcW, weights, kh, kw, stride = 1, padding = 0, current_coef = 1, inhibitory = false }) {
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW ${srcH * srcW}`);
  }
  const outH = Math.floor((srcH + 2 * padding - kh) / stride) + 1;
  const outW = Math.floor((srcW + 2 * padding - kw) / stride) + 1;
  const outSize = outH * outW;
  const conn = kh * kw;
  if (weights.length !== outSize * conn) {
    throw new Error(`weights length ${weights.length} != out_size*conn ${outSize * conn}`);
  }
  const out = new Array(outSize).fill(0);
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      const j = oh * outW + ow;
      let sum = 0;
      for (let a = 0; a < kh; a++) {
        for (let b = 0; b < kw; b++) {
          const ih = oh * stride + a - padding;
          const iw = ow * stride + b - padding;
          const val = (ih < 0 || ih >= srcH || iw < 0 || iw >= srcW) ? 0 : pre_spike[ih * srcW + iw];
          sum += val * weights[j * conn + a * kw + b];
        }
      }
      out[j] = sum;
    }
  }
  const coef = current_coef * currentType(inhibitory);
  return out.map((x) => coef * x);
}

// LateralDendriticInput: I = F.conv3d(pre_spike.view(1,1,1,H,W), weights(1,1,1,kh,kw), padding).view(-1)
// padding = (kh-1)//2 per spatial dim (odd kernel). Cross-correlation, stride 1.
export function lateralDendriticInput({ pre_spike, srcH, srcW, weights, kh, kw, current_coef = 1, inhibitory = false }) {
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW ${srcH * srcW}`);
  }
  if (weights.length !== kh * kw) {
    throw new Error(`weights length ${weights.length} != kh*kw ${kh * kw}`);
  }
  const padH = (kh - 1) >> 1;
  const padW = (kw - 1) >> 1;
  const out = new Array(srcH * srcW).fill(0);
  for (let oh = 0; oh < srcH; oh++) {
    for (let ow = 0; ow < srcW; ow++) {
      let sum = 0;
      for (let a = 0; a < kh; a++) {
        for (let b = 0; b < kw; b++) {
          const ih = oh + a - padH;
          const iw = ow + b - padW;
          const val = (ih < 0 || ih >= srcH || iw < 0 || iw >= srcW) ? 0 : pre_spike[ih * srcW + iw];
          sum += val * weights[a * kw + b];
        }
      }
      out[oh * srcW + ow] = sum;
    }
  }
  const coef = current_coef * currentType(inhibitory);
  return out.map((x) => coef * x);
}
