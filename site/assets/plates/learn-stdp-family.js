// Plate: STDP rule family across synaptic connection topologies.
// Covers all 13 CoNeX learning classes:
// SimpleSTDP, SparseSTDP, One2OneSTDP, Conv2dSTDP, Local2dSTDP,
// SimpleiSTDP, One2OneiSTDP, SparseiSTDP,
// SimpleRSTDP, One2OneRSTDP, SparseRSTDP, Conv2dRSTDP, Local2dRSTDP.

import * as M from '../js/models/learning.js';
import * as A from '../js/anim.js';

export const meta = {
  id: 'learn-stdp-family',
  title: 'STDP rule family',
  conex: [
    'SimpleSTDP', 'SparseSTDP', 'One2OneSTDP', 'Conv2dSTDP', 'Local2dSTDP',
    'SimpleiSTDP', 'One2OneiSTDP', 'SparseiSTDP',
    'SimpleRSTDP', 'One2OneRSTDP', 'SparseRSTDP', 'Conv2dRSTDP', 'Local2dRSTDP',
  ],
  source: 'conex/behaviors/synapses/learning.py',
};

// Generates reproducible Bernoulli spike trains from a seed for 16 pre and 4 post neurons over T steps.
function generateSpikeTrains(seed, steps = 40) {
  let s = (seed >>> 0) || 1;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const preSpikes16 = [];
  const postSpikes4 = [];
  for (let t = 0; t < steps; t++) {
    const preRow = [];
    for (let i = 0; i < 16; i++) {
      preRow.push(rand() < 0.28 ? 1 : 0);
    }
    preSpikes16.push(preRow);

    const postRow = [];
    for (let j = 0; j < 4; j++) {
      postRow.push(rand() < 0.32 ? 1 : 0);
    }
    postSpikes4.push(postRow);
  }
  return { preSpikes16, postSpikes4 };
}

// Build topological metadata and initial weights for each connection type
function buildTopology(connection, w0 = 0.5) {
  switch (connection) {
    case 'Sparse': {
      // 8 non-zero connections between 4 pre and 4 post neurons
      const src_idx = [0, 1, 0, 3, 0, 0, 2, 3];
      const dst_idx = [0, 0, 1, 1, 2, 3, 3, 3];
      return {
        is_sparse: true,
        src_idx,
        dst_idx,
        initial_weights: new Array(src_idx.length).fill(w0),
        rows: 4,
        cols: 4,
        rowLabel: 'post neuron',
        colLabel: 'pre neuron',
        toMatrix: (w) => {
          const mat = new Array(16).fill(0);
          for (let k = 0; k < src_idx.length; k++) {
            mat[src_idx[k] * 4 + dst_idx[k]] = w[k];
          }
          return mat;
        },
      };
    }
    case 'One2One': {
      // 4 one-to-one connections
      return {
        is_121: true,
        initial_weights: new Array(4).fill(w0),
        rows: 1,
        cols: 4,
        rowLabel: 'connection',
        colLabel: 'pre / post neuron index',
        toMatrix: (w) => [...w],
      };
    }
    case 'Conv2d': {
      // 3x3 shared kernel sweeping 4x4 input to 2x2 output
      return {
        is_conv: true,
        kh: 3,
        kw: 3,
        stride: 1,
        padding: 0,
        srcH: 4,
        srcW: 4,
        dstH: 2,
        dstW: 2,
        initial_weights: new Array(9).fill(w0),
        rows: 3,
        cols: 3,
        rowLabel: 'kernel row',
        colLabel: 'kernel col',
        toMatrix: (w) => [...w],
      };
    }
    case 'Local2d': {
      // 4 distinct 3x3 receptive fields for 4 destination neurons
      return {
        is_local: true,
        kh: 3,
        kw: 3,
        stride: 1,
        padding: 0,
        srcH: 4,
        srcW: 4,
        dstH: 2,
        dstW: 2,
        initial_weights: new Array(36).fill(w0),
        rows: 4,
        cols: 9,
        rowLabel: 'post neuron',
        colLabel: 'receptive field weight index (0..8)',
        toMatrix: (w) => [...w],
      };
    }
    case 'Simple':
    default: {
      // 4x4 dense matrix
      return {
        initial_weights: new Array(16).fill(w0),
        rows: 4,
        cols: 4,
        rowLabel: 'post neuron',
        colLabel: 'pre neuron',
        toMatrix: (w) => [...w],
      };
    }
  }
}

function mixHex(a, b, t) {
  const parse = (s) => {
    const m = String(s).match(/^#([0-9a-f]{6})$/i);
    if (!m) return [128, 128, 128];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const pa = parse(a), pb = parse(b);
  const clamped = Math.max(0, Math.min(1, t));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * clamped));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function drawHeatmap(canvasObj, ui, matrix, rows, cols, { rowLabel, colLabel, runMin, runMax, curStep }) {
  const { ctx, width, height } = canvasObj;
  ctx.clearRect(0, 0, width, height);

  const pal = {
    paper: ui.color('paper'),
    ink: ui.color('ink'),
    muted: ui.color('muted'),
    rule: ui.color('rule'),
    c1: ui.color('c1'),
  };

  const M = { left: 86, right: 80, top: 26, bottom: 42 };
  const plotW = Math.max(20, width - M.left - M.right);
  const plotH = Math.max(20, height - M.top - M.bottom);

  const cw = plotW / cols;
  const ch = plotH / rows;

  // Title
  ctx.fillStyle = pal.ink;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`Synaptic weights at step ${curStep}`, M.left, 6);

  // Draw cells
  const range = (runMax - runMin) || 1e-6;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = matrix[r * cols + c];
      const frac = Math.max(0, Math.min(1, (val - runMin) / range));
      ctx.fillStyle = mixHex(pal.paper, pal.c1, frac);
      ctx.fillRect(M.left + c * cw, M.top + r * ch, cw, ch);
      ctx.strokeStyle = pal.rule;
      ctx.strokeRect(M.left + c * cw, M.top + r * ch, cw, ch);
    }
  }

  // Border
  ctx.strokeStyle = pal.ink;
  ctx.lineWidth = 1;
  ctx.strokeRect(M.left, M.top, plotW, plotH);

  // Row tick labels
  ctx.fillStyle = pal.muted;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) {
    const y = M.top + (r + 0.5) * ch;
    ctx.fillText(String(r), M.left - 6, y);
  }

  // Row axis title
  ctx.fillStyle = pal.ink;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.save();
  ctx.translate(16, M.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(rowLabel, 0, 0);
  ctx.restore();

  // Column tick labels
  ctx.fillStyle = pal.muted;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let c = 0; c < cols; c++) {
    const x = M.left + (c + 0.5) * cw;
    ctx.fillText(String(c), x, M.top + plotH + 4);
  }

  // Column axis title
  ctx.fillStyle = pal.ink;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(colLabel, M.left + plotW / 2, height - 2);

  // Colour bar
  const barX = width - M.right + 20;
  const barW = 12;
  const barY = M.top;
  const barH = plotH;

  const grad = ctx.createLinearGradient ? ctx.createLinearGradient(0, barY + barH, 0, barY) : null;
  if (grad && typeof grad.addColorStop === 'function') {
    grad.addColorStop(0, pal.paper);
    grad.addColorStop(1, pal.c1);
    ctx.fillStyle = grad;
  } else {
    ctx.fillStyle = pal.c1;
  }
  ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = pal.ink;
  ctx.strokeRect(barX, barY, barW, barH);

  // Colour bar labels
  ctx.fillStyle = pal.muted;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'left';

  ctx.textBaseline = 'top';
  ctx.fillText(runMax.toFixed(3), barX + barW + 5, barY);

  ctx.textBaseline = 'bottom';
  ctx.fillText(runMin.toFixed(3), barX + barW + 5, barY + barH);

  const mid = (runMin + runMax) / 2;
  ctx.textBaseline = 'middle';
  ctx.fillText(mid.toFixed(3), barX + barW + 5, barY + barH / 2);

  ctx.fillStyle = pal.ink;
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('weight', barX + barW / 2 + 10, barY - 4);
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const hero = ui.hero(figures, { height: 300 });

  const state = {
    rule: 'STDP',
    connection: 'Simple',
    a_plus: 0.02,
    a_minus: 0.01,
    lr: 0.001,
    rho: 10.0,
    tau_s: 10.0,
    tau_c: 15.0,
    payoff: 1.0,
    tau_dopamine: 10.0,
    w_min: 0.0,
    w_max: 1.0,
    bound: 'no_bound',
    seed: 42,
    w0: 0.5,
    steps: 40,
  };

  let player = null;
  let cachedSimulation = null;
  let cachedTopology = null;
  let cachedPreSpikes = null;
  let cachedPostSpikes = null;
  let cachedRunMin = 0.45;
  let cachedRunMax = 0.55;
  let currentStep = state.steps;

  // 1. Input spike raster figure
  const rasterFig = ui.raster(figures, {
    height: 125,
    xLabel: 'time step',
    xUnit: 'steps',
  });

  const rasterCaption = document.createElement('figcaption');
  rasterCaption.style.cssText = 'font-size:12px;font-style:italic;color:var(--muted,#5f6b6d);margin:2px 0 10px 48px;line-height:1.4;';
  figures.appendChild(rasterCaption);

  // 2. Weights heatmap on canvas with labeled axes and colour scale
  const heatCanvas = ui.canvas(figures, { height: 230 });

  // 3. Mean weight plot over time
  const meanPlot = ui.plot(figures, {
    height: 180,
    xLabel: 'time step',
    xUnit: 'steps',
    yLabel: 'mean weight',
    yUnit: 'a.u.',
  });

  function resolveClassName() {
    let conn = state.connection;
    if (state.rule === 'iSTDP' && (conn === 'Conv2d' || conn === 'Local2d')) {
      conn = 'Simple';
    }
    if (state.rule === 'STDP') return `${conn}STDP`;
    if (state.rule === 'iSTDP') return `${conn}iSTDP`;
    return `${conn}RSTDP`;
  }

  function resolveDendriticInputClass(conn) {
    switch (conn) {
      case 'Sparse': return 'SparseDendriticInput';
      case 'One2One': return 'One2OneDendriticInput';
      case 'Conv2d': return 'Conv2dDendriticInput';
      case 'Local2d': return 'Local2dDendriticInput';
      case 'Simple':
      default: return 'SimpleDendriticInput';
    }
  }

  function recompute() {
    const className = resolveClassName();
    const topology = buildTopology(state.connection, state.w0);
    cachedTopology = topology;

    const { preSpikes16, postSpikes4 } = generateSpikeTrains(state.seed, state.steps);
    const pre_spikes = (topology.is_conv || topology.is_local)
      ? preSpikes16
      : preSpikes16.map((row) => row.slice(0, 4));
    const post_spikes = postSpikes4;

    cachedPreSpikes = pre_spikes;
    cachedPostSpikes = post_spikes;

    const params = {
      a_plus: state.a_plus,
      a_minus: state.a_minus,
      lr: state.lr,
      rho: state.rho,
      tau_s: state.tau_s,
      tau_c: state.tau_c,
      tau_dopamine: state.tau_dopamine,
      payoff: state.payoff,
      w_min: state.w_min,
      w_max: state.w_max,
      positive_bound: state.bound,
      negative_bound: state.bound,
    };

    cachedSimulation = M.simulateLearningTrace({
      className,
      params,
      meta: topology,
      pre_spikes,
      post_spikes,
    });

    let rMin = Infinity;
    let rMax = -Infinity;
    for (const stepW of cachedSimulation.history_weights) {
      for (const w of stepW) {
        if (w < rMin) rMin = w;
        if (w > rMax) rMax = w;
      }
    }
    if (!Number.isFinite(rMin) || rMin === rMax) {
      rMin = state.w0 - 0.05;
      rMax = state.w0 + 0.05;
    }
    cachedRunMin = rMin;
    cachedRunMax = rMax;

    if (topology.is_conv || topology.is_local) {
      rasterCaption.textContent =
        'Spike raster: rows 0–15 are pre neurons (indices 0–15, 4×4 input); rows 16–19 are post neurons (indices 0–3, 2×2 output).';
    } else {
      rasterCaption.textContent =
        'Spike raster: rows 0–3 are pre neurons (indices 0–3); rows 4–7 are post neurons (indices 0–3).';
    }
  }

  // Hero: the weight matrix / kernel as a grid of synapses whose size and colour
  // morph with the weights; pre/post spikes flash on their rows/cols; for Conv2d
  // a single kernel is shown sliding across the 4x4 input.
  function drawHero(t) {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c5 = ui.color('c5');
    const ink = ui.color('ink'), muted = ui.color('muted'), rule = ui.color('rule');
    ctx.clearRect(0, 0, W, H);
    const topo = cachedTopology;
    const weights = cachedSimulation.history_weights[t];
    const preRow = cachedPreSpikes[t];
    const postRow = cachedPostSpikes[t];
    const range = (cachedRunMax - cachedRunMin) || 1e-6;

    const isConv = topo.is_conv;
    const isLocal = topo.is_local;

    if (isConv || isLocal) {
      // input 4x4 grid (left) with the kernel window (right)
      const cell = Math.min(W * 0.16, H * 0.22);
      const gx0 = W * 0.06, gy0 = (H - 4 * cell) / 2;
      const nPre = cachedPreSpikes[0].length; // 16
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        const i = r * 4 + c;
        const on = preRow[i] === 1;
        ctx.fillStyle = on ? c2 : rule;
        ctx.globalAlpha = on ? 1 : 0.4;
        ctx.fillRect(gx0 + c * cell, gy0 + r * cell, cell, cell);
      }
      ctx.globalAlpha = 1;
      // the sliding kernel window (Conv2d: one shared kernel; Local2d: the
      // kernel of the output position the window covers)
      const kh = topo.kh, kw = topo.kw;
      const win = Math.floor(t / 2) % 2; // window slides one cell every 2 steps
      const wx0 = gx0 + win * cell, wy0 = gy0 + win * cell;
      ctx.strokeStyle = c5; ctx.lineWidth = 3;
      ctx.strokeRect(wx0, wy0, kh * cell, kw * cell);
      // the kernel weights as a small grid (right)
      const kcell = cell * 0.8;
      const kx0 = W * 0.62, ky0 = (H - kh * kcell) / 2;
      for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) {
        const wv = isLocal ? weights[(t % 4) * kh * kw + a * kw + b] : weights[a * kw + b];
        const frac = A.stdpFamCellFrac(wv, cachedRunMin, cachedRunMax);
        ctx.fillStyle = mixHex(ui.color('paper'), c1, frac);
        ctx.fillRect(kx0 + b * kcell, ky0 + a * kcell, kcell, kcell);
        ctx.strokeStyle = rule; ctx.lineWidth = 1;
        ctx.strokeRect(kx0 + b * kcell, ky0 + a * kcell, kcell, kcell);
      }
      ctx.fillStyle = ink; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('4x4 input (spikes flash)', gx0 + 2 * cell, H * 0.94);
      ctx.fillText(isLocal ? 'local kernel' : 'shared kernel', kx0 + (kw * kcell) / 2, H * 0.94);
      return;
    }

    // matrix / diagonal layouts: rows = post, cols = pre
    const rows = topo.rows, cols = topo.cols;
    const mat = topo.toMatrix(weights); // proper rows x cols matrix (sparse -> dense)
    const cell = Math.min(W * 0.12, H * 0.16);
    const gx0 = W * 0.28, gy0 = (H - rows * cell) / 2;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const wv = mat[r * cols + c];
      const frac = A.stdpFamCellFrac(wv, cachedRunMin, cachedRunMax);
      const size = cell * (0.35 + 0.65 * frac);
      const cx = gx0 + (c + 0.5) * cell, cy = gy0 + (r + 0.5) * cell;
      ctx.fillStyle = mixHex(ui.color('paper'), c1, frac);
      ctx.fillRect(cx - size / 2, cy - size / 2, size, size);
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
    }
    // pre spikes flash on their columns, post spikes on their rows
    ctx.fillStyle = c2;
    for (let c = 0; c < cols; c++) if (preRow[c] === 1) {
      ctx.globalAlpha = 0.5;
      ctx.fillRect(gx0 + c * cell, gy0 - 6, cell, 4);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = c5;
    for (let r = 0; r < rows; r++) {
      if (postRow[r] === 1) {
        ctx.globalAlpha = 0.5;
        ctx.fillRect(gx0 - 6, gy0 + r * cell, 4, cell);
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`${topo.rowLabel} (rows)  x  ${topo.colLabel} (cols)`, W / 2, H * 0.94);
  }

  function renderAt(step) {
    if (!cachedSimulation || !cachedTopology) recompute();
    const t = Math.max(0, Math.min(Math.floor(step), state.steps - 1));
    currentStep = t;

    // 1. Draw input spike raster (pre neurons on bottom rows, post neurons on top rows)
    const nPre = cachedPreSpikes[0].length;
    const nPost = cachedPostSpikes[0].length;
    const nTotal = nPre + nPost;
    const spikesList = [];
    for (let s = 0; s <= t; s++) {
      const pRow = cachedPreSpikes[s];
      for (let i = 0; i < nPre; i++) {
        if (pRow[i]) spikesList.push([s, i]);
      }
      const postRow = cachedPostSpikes[s];
      for (let j = 0; j < nPost; j++) {
        if (postRow[j]) spikesList.push([s, nPre + j]);
      }
    }
    rasterFig.draw(spikesList, nTotal, state.steps, { revealT: t });

    // 2. Draw weights heatmap with axes and colour bar
    const curWeights = cachedSimulation.history_weights[t];
    const heatValues = cachedTopology.toMatrix(curWeights);
    drawHeatmap(heatCanvas, ui, heatValues, cachedTopology.rows, cachedTopology.cols, {
      rowLabel: cachedTopology.rowLabel,
      colLabel: cachedTopology.colLabel,
      runMin: cachedRunMin,
      runMax: cachedRunMax,
      curStep: t,
    });

    // 3. Draw mean weight plot
    const xs = [];
    const ys = [];
    for (let i = 0; i < cachedSimulation.history_mean_weights.length; i++) {
      xs.push(i);
      ys.push(cachedSimulation.history_mean_weights[i]);
    }

    meanPlot.draw(
      [
        {
          x: xs,
          y: ys,
          label: 'mean weight',
          color: 'c1',
        },
      ],
      {
        xRange: [0, state.steps],
        revealX: t,
        hlines: [{ y: state.w0, label: 'w0' }],
      }
    );

    // 4. Hero: the weight matrix / kernel morphing with the weights
    drawHero(t);
    hero.narration(A.stdpFamNarration(state.rule, state.connection));
  }

  function update() {
    recompute();
    renderAt(currentStep);
  }

  heatCanvas.onResize(() => renderAt(currentStep));
  hero.onResize(() => renderAt(currentStep));

  // --- Controls ---
  const secSetup = ui.section(controls, 'Model selection');
  ui.select(secSetup, {
    key: 'rule',
    label: 'Learning rule',
    value: state.rule,
    options: [
      { value: 'STDP', label: 'STDP (classic)' },
      { value: 'iSTDP', label: 'iSTDP (inhibitory)' },
      { value: 'RSTDP', label: 'RSTDP (reward-modulated)' },
    ],
    onInput: (v) => {
      state.rule = v;
      if (v === 'iSTDP' && (state.connection === 'Conv2d' || state.connection === 'Local2d')) {
        state.connection = 'Simple';
        connSelect.set('Simple');
      }
      update();
    },
  });

  const connSelect = ui.select(secSetup, {
    key: 'connection',
    label: 'Topology',
    value: state.connection,
    options: [
      { value: 'Simple', label: 'Simple (dense 4x4)' },
      { value: 'Sparse', label: 'Sparse (density 0.5)' },
      { value: 'One2One', label: 'One2One (1-to-1)' },
      { value: 'Conv2d', label: 'Conv2d (shared 3x3)' },
      { value: 'Local2d', label: 'Local2d (local 3x3)' },
    ],
    onInput: (v) => {
      if (state.rule === 'iSTDP' && (v === 'Conv2d' || v === 'Local2d')) {
        state.connection = 'Simple';
        connSelect.set('Simple');
      } else {
        state.connection = v;
      }
      update();
    },
  });

  const secParams = ui.section(controls, 'Learning parameters');
  ui.slider(secParams, {
    key: 'a_plus',
    label: 'a_plus (STDP)',
    unit: 'a.u.',
    min: 0,
    max: 0.05,
    step: 0.001,
    value: state.a_plus,
    onInput: (v) => { state.a_plus = v; update(); },
  });
  ui.slider(secParams, {
    key: 'a_minus',
    label: 'a_minus (STDP)',
    unit: 'a.u.',
    min: 0,
    max: 0.05,
    step: 0.001,
    value: state.a_minus,
    onInput: (v) => { state.a_minus = v; update(); },
  });
  ui.slider(secParams, {
    key: 'lr',
    label: 'lr (iSTDP)',
    unit: 'a.u.',
    min: 0.0001,
    max: 0.005,
    step: 0.0001,
    value: state.lr,
    onInput: (v) => { state.lr = v; update(); },
  });
  ui.slider(secParams, {
    key: 'rho',
    label: 'rho (iSTDP)',
    unit: 'Hz',
    min: 1,
    max: 30,
    step: 1,
    value: state.rho,
    onInput: (v) => { state.rho = v; update(); },
  });
  ui.slider(secParams, {
    key: 'tau_c',
    label: 'tau_c (RSTDP)',
    unit: 'ms',
    min: 1,
    max: 50,
    step: 1,
    value: state.tau_c,
    onInput: (v) => { state.tau_c = v; update(); },
  });
  ui.slider(secParams, {
    key: 'payoff',
    label: 'payoff (RSTDP)',
    unit: 'a.u.',
    min: 0,
    max: 2,
    step: 0.1,
    value: state.payoff,
    onInput: (v) => { state.payoff = v; update(); },
  });
  ui.slider(secParams, {
    key: 'tau_s',
    label: 'tau_s (trace decay)',
    unit: 'ms',
    min: 1,
    max: 50,
    step: 1,
    value: state.tau_s,
    onInput: (v) => { state.tau_s = v; update(); },
  });
  ui.slider(secParams, {
    key: 'w0',
    label: 'initial weight w0',
    unit: 'a.u.',
    min: 0,
    max: 1,
    step: 0.05,
    value: state.w0,
    onInput: (v) => { state.w0 = v; update(); },
  });
  ui.number(secParams, {
    key: 'seed',
    label: 'random seed',
    unit: '',
    value: state.seed,
    onInput: (v) => { state.seed = Math.round(v); update(); },
  });

  // Animation player
  if (typeof ui.player === 'function') {
    player = ui.player(figures, {
      duration: state.steps,
      unit: 'steps',
      playSeconds: 5,
      loop: true,
      onFrame: (t) => renderAt(t),
    });
  }

  ui.onTheme(update);
  update();

  // Notes
  ui.note(
    notes,
    `CoNeX provides 13 synaptic learning rule classes across three rule paradigms (STDP, iSTDP, RSTDP) ` +
    `and five connection topologies (Simple, Sparse, One2One, Conv2d, Local2d). Classic STDP adapts weights ` +
    `from pre- and post-synaptic spike traces to implement associative plasticity. Symmetric iSTDP (Vogels et al. 2011) ` +
    `adjusts inhibitory strengths toward a target firing set-point rho. Reward-modulated STDP (RSTDP, Frémaux & Gerstner 2016) ` +
    `accumulates eligibility traces that are multiplicatively gated by extracellular dopamine.`
  );

  ui.equation(
    notes,
    `STDP: <var>dw</var> = <var>a</var><sub>+</sub>·<var>pre_trace</var>·<var>post_spike</var> − <var>a</var><sub>−</sub>·<var>pre_spike</var>·<var>post_trace</var><br>` +
    `iSTDP: <var>dw</var> = <var>lr</var>·[<var>pre_spike</var>·(<var>&alpha;</var> − <var>post_trace</var>)·<var>sign</var> + <var>pre_trace</var>·<var>post_spike</var>]<br>` +
    `RSTDP: <var>c</var><sub>t+1</sub> = <var>c</var><sub>t</sub>(1 − 1/<var>&tau;</var><sub>c</sub>) + <var>dw</var><sub>stdp</sub>, &nbsp; <var>&Delta;w</var> = <var>c</var>·<var>dopamine</var>`
  );

  ui.note(
    notes,
    `<strong>Topology breakdown (weights and sharing in CoNeX):</strong><br><br>` +
    `<strong>Simple (dense):</strong> A full dense matrix of size <code>n_src &times; n_dst</code> ` +
    `instantiates every pairwise connection (<code>dendrites.py:111-135</code>). In <code>SimpleSTDP</code>, ` +
    `updates are computed via outer products <code>torch.outer(pre_spike, post_trace)</code> and ` +
    `<code>torch.outer(pre_trace, post_spike)</code> (<code>learning.py:101-113</code>). Every connection has its ` +
    `own independent weight; no weights are shared across synapses.<br><br>` +
    `<strong>Sparse:</strong> Only instantiated connections exist, stored as a 1D tensor of non-zero entries ` +
    `<code>synapse.weights.values()[:]</code> indexed by <code>src_idx</code> and <code>dst_idx</code> ` +
    `(<code>dendrites.py:59-83</code>, <code>learning.py:131</code>). Non-existent connections have no weights and ` +
    `receive zero updates (<code>learning.py:130-149</code>). Each existing connection updates independently; ` +
    `no weights are shared.<br><br>` +
    `<strong>One2One:</strong> Only diagonal synapses exist: presynaptic neuron <code>i</code> connects strictly ` +
    `to postsynaptic neuron <code>i</code>, requiring <code>src.size == dst.size</code> (<code>dendrites.py:84-110</code>). ` +
    `In <code>One2OneSTDP</code>, updates are 1D elementwise vectors along the diagonal (<code>learning.py:166-180</code>). ` +
    `No weights are shared.<br><br>` +
    `<strong>Conv2d:</strong> A single 2D kernel of shape <code>(1, 1, kh, kw)</code> is swept across all spatial ` +
    `receptive field positions (<code>dendrites.py:209-306</code>). In <code>Conv2dSTDP</code>, presynaptic spikes and traces ` +
    `are unfolded and multiplied against postsynaptic targets, then divided by ` +
    `<code>self.weight_divisor = dst_shape[2] * dst_shape[1]</code> (<code>learning.py:307, 345</code>). ` +
    `Weights are fully shared across all output positions; the single kernel's update is the spatial average over all receptive fields.<br><br>` +
    `<strong>Local2d:</strong> Each destination neuron <code>j</code> maintains its own dedicated receptive field kernel ` +
    `of size <code>kh &times; kw</code>, giving total weight shape <code>(1, out_size, connection_size)</code> ` +
    `(<code>dendrites.py:307-385</code>, <code>learning.py:353-393</code>). Presynaptic inputs are unfolded locally, but ` +
    `unlike Conv2d, weights are local to each destination position and updates are not divided by output size (<code>learning.py:395</code>). ` +
    `No weights are shared across positions.<br><br>` +
    `<em>Note on iSTDP:</em> CoNeX only implements Simple, Sparse, and One2One topologies for iSTDP ` +
    `(<code>SimpleiSTDP</code>, <code>SparseiSTDP</code>, <code>One2OneiSTDP</code>; lines 183-293); Conv2d and Local2d iSTDP do not exist.`
  );

  ui.note(
    notes,
    `ML analogy: STDP resembles contrastive Hebbian learning and local weight gradient descent; the analogy breaks because STDP operates ` +
    `on asynchronous spike event times rather than smooth continuous activation values.`
  );

  // CoNeX code snippet
  ui.code(notes, () => {
    const cls = resolveClassName();
    const dCls = resolveDendriticInputClass(state.connection);
    const isRSTDP = cls.includes('RSTDP');
    const isISTDP = cls.includes('iSTDP');

    let dendriteCall = `${dCls}(current_coef=1)`;
    if (state.connection === 'Conv2d' || state.connection === 'Local2d') {
      dendriteCall = `${dCls}(current_coef=1, stride=1, padding=0)`;
    }

    let learnCall = '';
    if (isISTDP) {
      learnCall = `${cls}(lr=${state.lr}, rho=${state.rho}, is_inhibitory=True)`;
    } else if (isRSTDP) {
      learnCall = `${cls}(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c}, w_min=${state.w_min}, w_max=${state.w_max})`;
    } else {
      learnCall = `${cls}(a_plus=${state.a_plus}, a_minus=${state.a_minus}, w_min=${state.w_min}, w_max=${state.w_max})`;
    }

    const imports = [cls, dCls, 'PreTrace', 'PostTrace', 'PreSpikeCatcher', 'PostSpikeCatcher'];
    if (isRSTDP) imports.push('Dopamine', 'Payoff');

    const lines = [
      `from conex import ${imports.join(', ')}`,
    ];
    if (isRSTDP) {
      lines.push(`Payoff(initial_payoff=${state.payoff})`);
      lines.push(`Dopamine(tau_dopamine=${state.tau_dopamine})`);
    }
    lines.push(learnCall);
    lines.push(dendriteCall);
    lines.push(`PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})`);
    return lines.join('\n');
  });

  return {
    destroy() {
      if (player && typeof player.pause === 'function') {
        player.pause();
      }
    },
  };
}
