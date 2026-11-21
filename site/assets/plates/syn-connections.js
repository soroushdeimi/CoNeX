// Plate: Dendritic connection types — weight patterns and the current each post neuron gets.
// Eased player (AMENDMENT 4.3): the connection pattern is drawn as lines; input
// spikes travel along the real connections as particles and fill the post-neuron
// current bars, tweened with ease-in-out over the step.
import * as M from '../js/models/plasticity.js';
import * as A from '../js/anim.js';

export const meta = {
  id: 'syn-connections',
  title: 'Dendritic connection types',
  conex: ['SimpleDendriticInput', 'SparseDendriticInput', 'One2OneDendriticInput', 'Conv2dDendriticInput', 'Local2dDendriticInput', 'LateralDendriticInput', 'AveragePool2D'],
  source: 'conex/behaviors/synapses/dendrites.py',
};

const H = 4, W = 4, N = 16;

function inputPattern(name) {
  const p = new Array(N).fill(0);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (name === 'checker') p[i] = (x + y) % 2 === 0 ? 1 : 0;
    else if (name === 'diagonal') p[i] = x === y ? 1 : 0;
    else if (name === 'all') p[i] = 1;
    else if (name === 'left') p[i] = x < 2 ? 1 : 0;
  }
  return p;
}

function buildWeights(type, state, rand) {
  if (type === 'dense' || type === 'sparse') {
    const nDst = 3;
    const w = new Array(N * nDst);
    for (let i = 0; i < N * nDst; i++) {
      const r = rand();
      w[i] = type === 'sparse' && r > state.density ? 0 : rand();
    }
    return { w, nDst };
  }
  if (type === 'one2one') return { w: Array.from({ length: N }, () => rand()), nDst: N };
  if (type === 'conv2d') return { w: Array.from({ length: state.kh * state.kw }, () => rand()), nDst: 4 };
  const outH = Math.floor((H - state.kh) / state.stride) + 1;
  const outW = Math.floor((W - state.kw) / state.stride) + 1;
  if (type === 'local2d') return { w: Array.from({ length: outH * outW * state.kh * state.kw }, () => rand()), nDst: outH * outW };
  if (type === 'lateral') return { w: Array.from({ length: state.kh * state.kw }, () => rand()), nDst: N };
  return { w: null, nDst: 4 }; // avgpool has no weights
}

function connectionPattern(type, state, weights) {
  if (type === 'dense' || type === 'sparse') return { rows: N, cols: 3, values: weights.w };
  if (type === 'one2one') {
    const v = new Array(N * N).fill(0);
    for (let i = 0; i < N; i++) v[i * N + i] = weights.w[i];
    return { rows: N, cols: N, values: v };
  }
  if (type === 'conv2d' || type === 'lateral') return { rows: state.kh, cols: state.kw, values: weights.w };
  if (type === 'local2d') return { rows: state.kh, cols: state.kw, values: weights.w.slice(0, state.kh * state.kw) };
  const v = new Array(N).fill(0);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) v[y * W + x] = Math.floor(y / 2) * 2 + Math.floor(x / 2);
  return { rows: H, cols: W, values: v };
}

function computeCurrent(type, state, weights, spikes) {
  const coef = state.inhibitory ? -1 : 1;
  if (type === 'dense' || type === 'sparse') return M.simpleDendriticInput({ pre_spike: spikes, weights: weights.w, nDst: 3, inhibitory: state.inhibitory });
  if (type === 'one2one') return M.one2oneDendriticInput({ pre_spike: spikes, weights: weights.w, inhibitory: state.inhibitory });
  if (type === 'conv2d') return M.conv2dDendriticInput({ pre_spike: spikes, srcH: H, srcW: W, weights: weights.w, kh: state.kh, kw: state.kw, stride: state.stride, padding: 0, inhibitory: state.inhibitory });
  if (type === 'local2d') return M.local2dDendriticInput({ pre_spike: spikes, srcH: H, srcW: W, weights: weights.w, kh: state.kh, kw: state.kw, stride: state.stride, padding: 0, inhibitory: state.inhibitory });
  if (type === 'lateral') return M.lateralDendriticInput({ pre_spike: spikes, srcH: H, srcW: W, weights: weights.w, kh: state.kh, kw: state.kw, inhibitory: state.inhibitory });
  if (type === 'avgpool') return M.averagePool2D({ pre_spike: spikes, srcH: H, srcW: W, dstH: 2, dstW: 2, inhibitory: state.inhibitory });
  return [];
}

// Per-post-neuron contribution of each pre neuron (the current is linear in the
// spikes, so a one-hot source gives that source's share).
function contributions(type, state, weights, spikes) {
  const nDst = computeCurrent(type, state, weights, new Array(N).fill(0)).length;
  const out = new Array(nDst).fill(0).map(() => new Array(N).fill(0));
  for (let s = 0; s < N; s++) {
    if (!spikes[s]) continue;
    const one = new Array(N).fill(0);
    one[s] = 1;
    const I = computeCurrent(type, state, weights, one);
    for (let d = 0; d < nDst; d++) out[d][s] = I[d];
  }
  return out;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = { type: 'dense', density: 0.5, kh: 3, kw: 3, stride: 1, pattern: 'checker', inhibitory: false, seed: 7 };

  const tryThis = document.createElement('p');
  tryThis.className = 'ui-trythis';
  tryThis.innerHTML = '<b>Try this:</b> switch <b>type</b> to sparse and lower <b>density</b>; watch fewer connections carry the current.';
  figures.appendChild(tryThis);

  const hero = ui.hero(figures, { height: 300 });
  const inputHeat = ui.heatmap(figures, { rows: H, cols: W, label: 'input pattern (pre spikes)' });
  const connHeat = ui.heatmap(figures, { rows: 16, cols: 16, label: 'connection pattern' });
  const currentPlot = ui.plot(figures, { height: 180, xLabel: 'post neuron', xUnit: 'index', yLabel: 'current I', yUnit: 'a.u.' });

  let data = null; // { spikes, conn, contrib, current, nDst, yMax }
  let curT = 0;
  let player = null;

  function compute() {
    const rand = M.mulberry32(state.seed);
    const spikes = inputPattern(state.pattern);
    const weights = buildWeights(state.type, state, rand);
    const conn = connectionPattern(state.type, state, weights);
    const contrib = contributions(state.type, state, weights, spikes);
    const nDst = contrib.length;
    const current = contrib.map((row) => row.reduce((a, b) => a + b, 0));
    const yMax = Math.max(0.1, ...current.map(Math.abs));
    data = { spikes, conn, contrib, current, nDst, yMin: -yMax, yMax };
  }

  // The real connection pattern as lines; spikes travel along them as particles
  // and the post-neuron current bars fill, all tweened by the eased player time.
  function drawHero(t) {
    const ctx = hero.ctx, Wd = hero.width, Hh = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, Wd, Hh);
    const nDst = data.nDst;
    const leftX = Wd * 0.5, rightX = Wd * 0.82;
    const py = (d) => (Hh * (d + 0.5)) / nDst;
    const iy = (i) => (Hh * (Math.floor(i / W) + 0.5)) / H;
    const ix = (i) => Wd * 0.18 - (i % W) * 8;
    const e = A.connProgress(t); // eased progress of the spike travel

    // connection lines (real pattern), thickness ~ contribution
    for (let d = 0; d < nDst; d++) for (let s = 0; s < N; s++) {
      if (!data.contrib[d][s]) continue;
      ctx.strokeStyle = muted; ctx.lineWidth = 0.5 + Math.abs(data.contrib[d][s]) * 2;
      ctx.globalAlpha = 0.3;
      ctx.beginPath(); ctx.moveTo(ix(s), iy(s)); ctx.lineTo(leftX, py(d)); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // input grid cells (pre neurons)
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = data.spikes[i] ? c2 : muted;
      ctx.globalAlpha = data.spikes[i] ? 1 : 0.3;
      ctx.fillRect(ix(i) - 3, iy(i) - 3, 6, 6);
    }
    ctx.globalAlpha = 1;
    // travelling spike particles along each real connection
    for (let d = 0; d < nDst; d++) for (let s = 0; s < N; s++) {
      if (!data.contrib[d][s]) continue;
      const x = ix(s) + (leftX - ix(s)) * e;
      const y = iy(s) + (py(d) - iy(s)) * e;
      ctx.fillStyle = c2; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    }
    // post neurons + current bars (fill with the eased progress)
    for (let d = 0; d < nDst; d++) {
      const val = data.current[d] * e;
      const bw = (val / data.yMax) * (Wd * 0.3);
      ctx.fillStyle = c1;
      ctx.fillRect(bw >= 0 ? rightX : rightX - Math.abs(bw), py(d) - 4, Math.abs(bw), 8);
      ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(rightX - 6, py(d), 4, 0, Math.PI * 2); ctx.fill();
    }
    // index labels: pre neurons (left), post neurons (right)
    ctx.fillStyle = muted; ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) ctx.fillText(String(i), ix(i) - 6, iy(i));
    ctx.textAlign = 'left';
    for (let d = 0; d < nDst; d++) ctx.fillText(String(d), rightX + 8, py(d));
    ctx.fillStyle = ink; ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText('pre', Wd * 0.02, Hh * 0.02);
    ctx.textAlign = 'right'; ctx.fillText('post', Wd * 0.98, Hh * 0.02);
    // weight colour bar (0..1) for the connection pattern
    const bx = Wd * 0.9, by = Hh * 0.5, bw2 = 10, bh = Hh * 0.4;
    const grad = typeof ctx.createLinearGradient === 'function' ? ctx.createLinearGradient(0, by + bh, 0, by) : null;
    if (grad && typeof grad.addColorStop === 'function') {
      grad.addColorStop(0, ui.color('paper')); grad.addColorStop(1, c1); ctx.fillStyle = grad;
    } else ctx.fillStyle = c1;
    ctx.fillRect(bx, by, bw2, bh);
    ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw2, bh);
    ctx.fillStyle = muted; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('1', bx + bw2 + 3, by);
    ctx.fillText('0', bx + bw2 + 3, by + bh);
  }

  function renderAt(t) {
    if (!data) return;
    curT = t;
    const e = ui.ease(t); // tween the current bars and the plot with the same easing
    inputHeat.draw(data.spikes, { min: 0, max: 1 });
    connHeat.draw(data.conn.values, { min: 0, max: 1 });
    const xs = data.current.map((_, i) => i);
    currentPlot.draw(
      [{ x: xs, y: data.current.map((v) => v * e), label: 'current', color: 'c1', style: 'bars' }],
      { xRange: [0, Math.max(1, data.nDst)], yRange: [data.yMin, data.yMax] }
    );
    drawHero(e);
    hero.narration(`spikes travel the ${state.type} connections and fill the post-neuron current bars`);
  }

  function recompute() {
    compute();
    if (player) { player.setDuration(1); renderAt(Math.min(curT, 1)); }
    else renderAt(1); // no player (checker): show the full current
  }

  const sec = ui.section(controls, 'Connection');
  ui.select(sec, {
    key: 'type', label: 'type', value: state.type,
    options: [
      { value: 'dense', label: 'dense (Simple)' },
      { value: 'sparse', label: 'sparse' },
      { value: 'one2one', label: 'one-to-one' },
      { value: 'conv2d', label: 'conv2d' },
      { value: 'local2d', label: 'local2d' },
      { value: 'lateral', label: 'lateral' },
      { value: 'avgpool', label: 'average pool' },
    ],
    onInput: (v) => { state.type = v; recompute(); },
  });
  ui.select(sec, {
    key: 'pattern', label: 'input pattern', value: state.pattern,
    options: [
      { value: 'checker', label: 'checkerboard' },
      { value: 'diagonal', label: 'diagonal' },
      { value: 'left', label: 'left half' },
      { value: 'all', label: 'all on' },
    ],
    onInput: (v) => { state.pattern = v; recompute(); },
  });
  ui.slider(sec, { key: 'density', label: 'density (sparse)', unit: 'a.u.', min: 0.05, max: 1, step: 0.05, value: state.density, onInput: (v) => { state.density = v; recompute(); } });
  ui.slider(sec, { key: 'kh', label: 'kernel height', unit: 'px', min: 1, max: 3, step: 2, value: state.kh, onInput: (v) => { state.kh = v; recompute(); } });
  ui.slider(sec, { key: 'kw', label: 'kernel width', unit: 'px', min: 1, max: 3, step: 2, value: state.kw, onInput: (v) => { state.kw = v; recompute(); } });
  ui.slider(sec, { key: 'stride', label: 'stride', unit: 'px', min: 1, max: 2, step: 1, value: state.stride, onInput: (v) => { state.stride = v; recompute(); } });
  ui.toggle(sec, { key: 'inhibitory', label: 'inhibitory source', value: state.inhibitory, onInput: (v) => { state.inhibitory = v; recompute(); } });
  ui.number(sec, { key: 'seed', label: 'weight seed', unit: '', value: state.seed, onInput: (v) => { state.seed = v; recompute(); } });

  recompute();
  player = (typeof ui.player === 'function') ? ui.player(controls, {
    duration: 1, unit: 'a.u.', playSeconds: 3, loop: true,
    onFrame: (t) => renderAt(t),
  }) : null;
  if (player) player.seek(0);
  hero.onResize(() => renderAt(curT));
  ui.onTheme(() => renderAt(curT));

  ui.note(notes, `These are the ways CoNeX turns pre-synaptic spikes into a post-synaptic current <var>I</var>. <b>Dense</b> connects every pre to every post; <b>sparse</b> keeps only a fraction (density); <b>one-to-one</b> pairs each pre with one post; <b>conv2d</b> and <b>lateral</b> share a small kernel across space; <b>local2d</b> gives each post its own kernel over its local patch; <b>average pool</b> downsamples by averaging. The sign of the current flips for an inhibitory source. The hero draws the real connection pattern as lines; each active pre neuron's spike travels along its connection as a particle and fills the post-neuron current bar.`);
  ui.equation(notes, `<var>I</var> = current_coef · current_type · calculate_input, &nbsp; current_type = −1 if inhibitory else 1`);
  ui.note(notes, `The current is dimensionless (a.u.); the grid is 4x4 pixels. ML analogy: these are exactly the weight layouts of a dense layer, a sparse mask, an identity map, a convolution, a local (unshared) convolution, and a pooling layer. The analogy breaks because here the "input" is a spike (0/1), not a continuous activation, and the current feeds a spiking neuron rather than a linear unit.`);
  ui.code(notes, () => {
    const cls = { dense: 'SimpleDendriticInput', sparse: 'SparseDendriticInput', one2one: 'One2OneDendriticInput', conv2d: 'Conv2dDendriticInput', local2d: 'Local2dDendriticInput', lateral: 'LateralDendriticInput', avgpool: 'AveragePool2D' }[state.type];
    const extra = state.type === 'conv2d' || state.type === 'local2d' ? `, stride=${state.stride}, padding=0` : '';
    return `from conex import ${cls}\n${cls}(current_coef=1${extra})`;
  });
  return { destroy() { if (player) player.pause(); } };
}
