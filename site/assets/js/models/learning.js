// Pure math for the learning rules area.
// Mirrors CoNeX Python (conex/behaviors/synapses/learning.py and conex/behaviors/synapses/specs.py).
// Covers all 13 synaptic learning rule classes:
// SimpleSTDP, SparseSTDP, One2OneSTDP, Conv2dSTDP, Local2dSTDP,
// SimpleiSTDP, One2OneiSTDP, SparseiSTDP,
// SimpleRSTDP, One2OneRSTDP, SparseRSTDP, Conv2dRSTDP, Local2dRSTDP,
// and the three synaptic weight bounding functions: soft_bound, hard_bound, no_bound.
// No DOM, no external dependencies. All time in ms, weights dimensionless (a.u.).

function checkFinite(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number, got ${String(value)}`);
  }
}

// --- Weight bounds (conex/behaviors/synapses/learning.py lines 15-27) ---

export const softBound = (w, w_min, w_max) => (w - w_min) * (w_max - w);
export const hardBound = (w, w_min, w_max) => (w > w_min && w < w_max ? 1 : 0);
export const noBound = () => 1;

export const BOUNDS = {
  soft_bound: softBound,
  hard_bound: hardBound,
  no_bound: noBound,
};

export function boundFn(name) {
  if (typeof name === 'function') return name;
  if (name === null || name === undefined) return noBound;
  if (!BOUNDS[name]) {
    throw new Error(`unknown bound "${name}", expected one of ${Object.keys(BOUNDS).join(', ')}`);
  }
  return BOUNDS[name];
}

// Trajectory under repeated potentiation then depression
export function simulateBoundTrajectory({
  w0,
  dw_plus,
  dw_minus,
  n_plus,
  n_minus,
  w_min = 0.0,
  w_max = 1.0,
  bound = 'no_bound',
} = {}) {
  checkFinite('w0', w0);
  checkFinite('dw_plus', dw_plus);
  checkFinite('dw_minus', dw_minus);
  checkFinite('n_plus', n_plus);
  checkFinite('n_minus', n_minus);
  checkFinite('w_min', w_min);
  checkFinite('w_max', w_max);
  if (w_min >= w_max) {
    throw new Error(`w_min (${w_min}) must be strictly less than w_max (${w_max})`);
  }
  const fn = boundFn(bound);
  let w = w0;
  const traj = [w];
  for (let t = 0; t < n_plus; t++) {
    const factor = fn(w, w_min, w_max);
    w = w + dw_plus * factor;
    traj.push(w);
  }
  for (let t = 0; t < n_minus; t++) {
    const factor = fn(w, w_min, w_max);
    w = w - dw_minus * factor;
    traj.push(w);
  }
  return traj;
}

// --- Spike traces (conex/behaviors/synapses/specs.py PreTrace/PostTrace) ---

export function stepTrace(trace, spike, { tau_s, spike_scale = 1.0, dt = 1.0 } = {}) {
  checkFinite('tau_s', tau_s);
  checkFinite('dt', dt);
  if (tau_s <= 0) throw new Error(`tau_s must be > 0, got ${tau_s}`);
  if (dt <= 0) throw new Error(`dt must be > 0, got ${dt}`);
  const s = spike ? 1 : 0;
  const updated = trace + s * spike_scale;
  return updated - (updated / tau_s) * dt;
}

// --- Neuromodulators & RSTDP (conex/behaviors/network/neuromodulators.py) ---

export function dopamineStep({ dopamine_concentration, tau_dopamine, payoff }) {
  checkFinite('tau_dopamine', tau_dopamine);
  checkFinite('payoff', payoff);
  if (tau_dopamine <= 0) throw new Error(`tau_dopamine must be > 0, got ${tau_dopamine}`);
  const dd_dt = -(dopamine_concentration / tau_dopamine) + payoff;
  return dopamine_concentration + dd_dt;
}

export function rstdpStep({ c, dw, tau_c, dopamine_concentration }) {
  checkFinite('tau_c', tau_c);
  checkFinite('dopamine_concentration', dopamine_concentration);
  if (tau_c <= 0) throw new Error(`tau_c must be > 0, got ${tau_c}`);
  if (c.length !== dw.length) {
    throw new Error(`eligibility trace length ${c.length} != dw length ${dw.length}`);
  }
  const cNew = new Array(c.length);
  const dwApplied = new Array(c.length);
  for (let i = 0; i < c.length; i++) {
    cNew[i] = c[i] - c[i] / tau_c + dw[i];
    dwApplied[i] = cNew[i] * dopamine_concentration;
  }
  return { c: cNew, dwApplied };
}

// --- STDP compute_dw variants ---

// 1. SimpleSTDP (dense connection)
export function simpleStdpDw({
  weights,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  a_plus,
  a_minus,
  w_min = 0.0,
  w_max = 1.0,
  positive_bound = 'no_bound',
  negative_bound = 'no_bound',
}) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  const nSrc = pre_spike.length;
  const nDst = post_spike.length;
  if (weights.length !== nSrc * nDst) {
    throw new Error(`weights length ${weights.length} != nSrc*nDst (${nSrc * nDst})`);
  }
  const p = boundFn(positive_bound);
  const n = boundFn(negative_bound);
  const dw = new Array(nSrc * nDst).fill(0);
  for (let s = 0; s < nSrc; s++) {
    const sSpk = pre_spike[s] ? 1 : 0;
    const sTr = pre_trace[s];
    for (let d = 0; d < nDst; d++) {
      const dSpk = post_spike[d] ? 1 : 0;
      const dTr = post_trace[d];
      const idx = s * nDst + d;
      const w = weights[idx];
      const dw_plus = sTr * dSpk * a_plus * p(w, w_min, w_max);
      const dw_minus = sSpk * dTr * a_minus * n(w, w_min, w_max);
      dw[idx] = dw_plus - dw_minus;
    }
  }
  return dw;
}

// 2. SparseSTDP (sparse connection with src_idx and dst_idx)
export function sparseStdpDw({
  weights,
  src_idx,
  dst_idx,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  a_plus,
  a_minus,
  w_min = 0.0,
  w_max = 1.0,
  positive_bound = 'no_bound',
  negative_bound = 'no_bound',
}) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  const nnz = weights.length;
  if (src_idx.length !== nnz || dst_idx.length !== nnz) {
    throw new Error(`sparse index lengths (src: ${src_idx.length}, dst: ${dst_idx.length}) != weights length (${nnz})`);
  }
  const p = boundFn(positive_bound);
  const n = boundFn(negative_bound);
  const dw = new Array(nnz).fill(0);
  for (let k = 0; k < nnz; k++) {
    const s = src_idx[k];
    const d = dst_idx[k];
    const sSpk = pre_spike[s] ? 1 : 0;
    const dSpk = post_spike[d] ? 1 : 0;
    const sTr = pre_trace[s];
    const dTr = post_trace[d];
    const w = weights[k];
    const dw_plus = sTr * dSpk * a_plus * p(w, w_min, w_max);
    const dw_minus = sSpk * dTr * a_minus * n(w, w_min, w_max);
    dw[k] = dw_plus - dw_minus;
  }
  return dw;
}

// 3. One2OneSTDP (one-to-one connection where nSrc == nDst)
export function one2oneStdpDw({
  weights,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  a_plus,
  a_minus,
  w_min = 0.0,
  w_max = 1.0,
  positive_bound = 'no_bound',
  negative_bound = 'no_bound',
}) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  const n = weights.length;
  if (pre_spike.length !== n || post_spike.length !== n) {
    throw new Error(`pre/post spike length (${pre_spike.length}, ${post_spike.length}) != weights length (${n})`);
  }
  const p = boundFn(positive_bound);
  const nFn = boundFn(negative_bound);
  const dw = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const sSpk = pre_spike[i] ? 1 : 0;
    const dSpk = post_spike[i] ? 1 : 0;
    const sTr = pre_trace[i];
    const dTr = post_trace[i];
    const w = weights[i];
    const dw_plus = sTr * dSpk * a_plus * p(w, w_min, w_max);
    const dw_minus = sSpk * dTr * a_minus * nFn(w, w_min, w_max);
    dw[i] = dw_plus - dw_minus;
  }
  return dw;
}

// 4. Conv2dSTDP (2D convolutional connection with shared kernel)
export function conv2dStdpDw({
  weights,
  srcH,
  srcW,
  dstH,
  dstW,
  kh,
  kw,
  stride = 1,
  padding = 0,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  a_plus,
  a_minus,
  w_min = 0.0,
  w_max = 1.0,
  positive_bound = 'no_bound',
  negative_bound = 'no_bound',
}) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  if (weights.length !== kh * kw) {
    throw new Error(`conv2d kernel weights length ${weights.length} != kh*kw (${kh * kw})`);
  }
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW (${srcH * srcW})`);
  }
  if (post_spike.length !== dstH * dstW) {
    throw new Error(`post_spike length ${post_spike.length} != dstH*dstW (${dstH * dstW})`);
  }
  const p = boundFn(positive_bound);
  const n = boundFn(negative_bound);
  const divisor = dstH * dstW;

  const dw_plus_accum = new Array(kh * kw).fill(0);
  const dw_minus_accum = new Array(kh * kw).fill(0);

  for (let oh = 0; oh < dstH; oh++) {
    for (let ow = 0; ow < dstW; ow++) {
      const dIdx = oh * dstW + ow;
      const dSpk = post_spike[dIdx] ? 1 : 0;
      const dTr = post_trace[dIdx];

      for (let ki = 0; ki < kh; ki++) {
        for (let kj = 0; kj < kw; kj++) {
          const k = ki * kw + kj;
          const ih = oh * stride + ki - padding;
          const iw = ow * stride + kj - padding;
          if (ih >= 0 && ih < srcH && iw >= 0 && iw < srcW) {
            const sIdx = ih * srcW + iw;
            const sSpk = pre_spike[sIdx] ? 1 : 0;
            const sTr = pre_trace[sIdx];
            dw_minus_accum[k] += sSpk * dTr;
            dw_plus_accum[k] += sTr * dSpk;
          }
        }
      }
    }
  }

  const dw = new Array(kh * kw).fill(0);
  for (let k = 0; k < kh * kw; k++) {
    const w = weights[k];
    const dw_plus = dw_plus_accum[k] * p(w, w_min, w_max);
    const dw_minus = dw_minus_accum[k] * n(w, w_min, w_max);
    dw[k] = (dw_plus * a_plus - dw_minus * a_minus) / divisor;
  }
  return dw;
}

// 5. Local2dSTDP (2D local receptive fields with individual weights per patch)
export function local2dStdpDw({
  weights,
  srcH,
  srcW,
  dstH,
  dstW,
  kh,
  kw,
  stride = 1,
  padding = 0,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  a_plus,
  a_minus,
  w_min = 0.0,
  w_max = 1.0,
  positive_bound = 'no_bound',
  negative_bound = 'no_bound',
}) {
  checkFinite('a_plus', a_plus);
  checkFinite('a_minus', a_minus);
  const outSize = dstH * dstW;
  const connSize = kh * kw;
  if (weights.length !== outSize * connSize) {
    throw new Error(`local2d weights length ${weights.length} != outSize*connSize (${outSize * connSize})`);
  }
  if (pre_spike.length !== srcH * srcW) {
    throw new Error(`pre_spike length ${pre_spike.length} != srcH*srcW (${srcH * srcW})`);
  }
  if (post_spike.length !== outSize) {
    throw new Error(`post_spike length ${post_spike.length} != outSize (${outSize})`);
  }
  const p = boundFn(positive_bound);
  const n = boundFn(negative_bound);
  const dw = new Array(outSize * connSize).fill(0);

  for (let oh = 0; oh < dstH; oh++) {
    for (let ow = 0; ow < dstW; ow++) {
      const j = oh * dstW + ow;
      const dSpk = post_spike[j] ? 1 : 0;
      const dTr = post_trace[j];

      for (let ki = 0; ki < kh; ki++) {
        for (let kj = 0; kj < kw; kj++) {
          const k = ki * kw + kj;
          const wIdx = j * connSize + k;
          const w = weights[wIdx];
          const ih = oh * stride + ki - padding;
          const iw = ow * stride + kj - padding;

          let sSpk = 0;
          let sTr = 0;
          if (ih >= 0 && ih < srcH && iw >= 0 && iw < srcW) {
            const sIdx = ih * srcW + iw;
            sSpk = pre_spike[sIdx] ? 1 : 0;
            sTr = pre_trace[sIdx];
          }

          const dw_plus = dSpk * sTr * p(w, w_min, w_max);
          const dw_minus = dTr * sSpk * n(w, w_min, w_max);
          dw[wIdx] = dw_plus * a_plus - dw_minus * a_minus;
        }
      }
    }
  }
  return dw;
}

// --- iSTDP variants (conex/behaviors/synapses/learning.py SimpleiSTDP, One2OneiSTDP, SparseiSTDP) ---

export function istdpAlpha({ rho, alpha = null, tau_s }) {
  if (alpha !== null && alpha !== undefined) return alpha;
  checkFinite('rho', rho);
  checkFinite('tau_s', tau_s);
  return (2 * rho * tau_s) / 1000;
}

// 6. SimpleiSTDP
export function simpleIstdpDw({
  weights,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  rho,
  alpha = null,
  lr = 1e-5,
  is_inhibitory = true,
  tau_s,
}) {
  checkFinite('lr', lr);
  const a = istdpAlpha({ rho, alpha, tau_s });
  const change_sign = 1 - (is_inhibitory ? 1 : 0) * 2;
  const nSrc = pre_spike.length;
  const nDst = post_spike.length;
  if (weights.length !== nSrc * nDst) {
    throw new Error(`weights length ${weights.length} != nSrc*nDst (${nSrc * nDst})`);
  }
  const dw = new Array(nSrc * nDst).fill(0);
  for (let s = 0; s < nSrc; s++) {
    const sSpk = pre_spike[s] ? 1 : 0;
    const sTr = pre_trace[s];
    for (let d = 0; d < nDst; d++) {
      const dSpk = post_spike[d] ? 1 : 0;
      const dTr = post_trace[d];
      const pre_spike_changes = sSpk * (a - dTr) * change_sign;
      const post_spike_changes = sTr * dSpk;
      dw[s * nDst + d] = lr * (pre_spike_changes + post_spike_changes);
    }
  }
  return dw;
}

// 7. One2OneiSTDP
export function one2oneIstdpDw({
  weights,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  rho,
  alpha = null,
  lr = 1e-5,
  is_inhibitory = true,
  tau_s,
}) {
  checkFinite('lr', lr);
  const a = istdpAlpha({ rho, alpha, tau_s });
  const change_sign = 1 - (is_inhibitory ? 1 : 0) * 2;
  const n = weights.length;
  if (pre_spike.length !== n || post_spike.length !== n) {
    throw new Error(`pre/post spike length (${pre_spike.length}, ${post_spike.length}) != weights length (${n})`);
  }
  const dw = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const sSpk = pre_spike[i] ? 1 : 0;
    const dSpk = post_spike[i] ? 1 : 0;
    const sTr = pre_trace[i];
    const dTr = post_trace[i];
    const pre_spike_changes = sSpk * (a - dTr) * change_sign;
    const post_spike_changes = sTr * dSpk;
    dw[i] = lr * (pre_spike_changes + post_spike_changes);
  }
  return dw;
}

// 8. SparseiSTDP
export function sparseIstdpDw({
  weights,
  src_idx,
  dst_idx,
  pre_spike,
  post_spike,
  pre_trace,
  post_trace,
  rho,
  alpha = null,
  lr = 1e-5,
  is_inhibitory = true,
  tau_s,
}) {
  checkFinite('lr', lr);
  const a = istdpAlpha({ rho, alpha, tau_s });
  const change_sign = 1 - (is_inhibitory ? 1 : 0) * 2;
  const nnz = weights.length;
  if (src_idx.length !== nnz || dst_idx.length !== nnz) {
    throw new Error(`sparse index lengths (src: ${src_idx.length}, dst: ${dst_idx.length}) != weights length (${nnz})`);
  }
  const dw = new Array(nnz).fill(0);
  for (let k = 0; k < nnz; k++) {
    const s = src_idx[k];
    const d = dst_idx[k];
    const sSpk = pre_spike[s] ? 1 : 0;
    const dSpk = post_spike[d] ? 1 : 0;
    const sTr = pre_trace[s];
    const dTr = post_trace[d];
    const pre_spike_changes = sSpk * (a - dTr) * change_sign;
    const post_spike_changes = sTr * dSpk;
    dw[k] = lr * (pre_spike_changes + post_spike_changes);
  }
  return dw;
}

// --- Simulation orchestrator for all 13 classes ---

export const LEARNING_CLASSES = [
  'SimpleSTDP', 'SparseSTDP', 'One2OneSTDP', 'Conv2dSTDP', 'Local2dSTDP',
  'SimpleiSTDP', 'One2OneiSTDP', 'SparseiSTDP',
  'SimpleRSTDP', 'One2OneRSTDP', 'SparseRSTDP', 'Conv2dRSTDP', 'Local2dRSTDP',
];

export function computeDwForClass(className, args) {
  switch (className) {
    case 'SimpleSTDP':
    case 'SimpleRSTDP':
      return simpleStdpDw(args);
    case 'SparseSTDP':
    case 'SparseRSTDP':
      return sparseStdpDw(args);
    case 'One2OneSTDP':
    case 'One2OneRSTDP':
      return one2oneStdpDw(args);
    case 'Conv2dSTDP':
    case 'Conv2dRSTDP':
      return conv2dStdpDw(args);
    case 'Local2dSTDP':
    case 'Local2dRSTDP':
      return local2dStdpDw(args);
    case 'SimpleiSTDP':
      return simpleIstdpDw(args);
    case 'SparseiSTDP':
      return sparseIstdpDw(args);
    case 'One2OneiSTDP':
      return one2oneIstdpDw(args);
    default:
      throw new Error(`unknown learning class "${className}"`);
  }
}

export function simulateLearningTrace({
  className,
  params,
  meta,
  pre_spikes,
  post_spikes,
  dt = 1.0,
}) {
  if (!LEARNING_CLASSES.includes(className)) {
    throw new Error(`unknown learning class "${className}", expected one of ${LEARNING_CLASSES.join(', ')}`);
  }
  const isRSTDP = className.includes('RSTDP');
  const isISTDP = className.includes('iSTDP');

  let weights = [...meta.initial_weights];
  let c = new Array(weights.length).fill(0);
  let dopamine_concentration = params.payoff ?? 0.0;
  const tau_s = params.tau_s;

  const nPre = pre_spikes[0].length;
  const nPost = post_spikes[0].length;
  let preTrace = new Array(nPre).fill(0);
  let postTrace = new Array(nPost).fill(0);

  const steps = pre_spikes.length;
  const history_weights = [];
  const history_mean_weights = [];

  for (let t = 0; t < steps; t++) {
    const curPreSpikes = pre_spikes[t];
    const curPostSpikes = post_spikes[t];

    // 1. Advance traces
    for (let s = 0; s < nPre; s++) {
      preTrace[s] = stepTrace(preTrace[s], curPreSpikes[s], { tau_s, dt });
    }
    for (let d = 0; d < nPost; d++) {
      postTrace[d] = stepTrace(postTrace[d], curPostSpikes[d], { tau_s, dt });
    }

    // 2. Compute dw
    const dwArgs = {
      weights,
      pre_spike: curPreSpikes,
      post_spike: curPostSpikes,
      pre_trace: preTrace,
      post_trace: postTrace,
      ...params,
      ...meta,
    };
    const dw = computeDwForClass(className, dwArgs);

    // 3. Apply weight change (or RSTDP eligibility / dopamine modulation)
    if (isRSTDP) {
      dopamine_concentration = dopamineStep({
        dopamine_concentration,
        tau_dopamine: params.tau_dopamine,
        payoff: params.payoff,
      });
      const { c: cNew, dwApplied } = rstdpStep({
        c,
        dw,
        tau_c: params.tau_c,
        dopamine_concentration,
      });
      c = cNew;
      for (let i = 0; i < weights.length; i++) {
        weights[i] += dwApplied[i];
      }
    } else {
      for (let i = 0; i < weights.length; i++) {
        weights[i] += dw[i];
      }
    }

    const snap = [...weights];
    history_weights.push(snap);
    const mean = snap.reduce((acc, v) => acc + v, 0) / snap.length;
    history_mean_weights.push(mean);
  }

  return {
    history_weights,
    history_mean_weights,
    final_weights: history_weights[history_weights.length - 1],
  };
}
