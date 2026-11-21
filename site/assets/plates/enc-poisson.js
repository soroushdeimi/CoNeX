// Plate: Poisson & latency encoding of a 16x16 grayscale pattern.
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'enc-poisson',
  title: 'Poisson & latency encoding',
  conex: ['SimplePoisson', 'Poisson', 'Intensity2Latency'],
  source: 'conex/helpers/transforms/encoders.py',
};

// A digit-like "7" stroke on a 16x16 grid, values in [0,1]. Procedural, no data.
function pattern16() {
  const img = new Array(256).fill(0);
  for (let x = 0; x < 16; x++) { img[2 * 16 + x] = 1.0; img[3 * 16 + x] = 0.6; }
  for (let y = 2; y < 14; y++) {
    const x = Math.floor(13 - (y - 2) * 0.8);
    img[y * 16 + x] = 1.0;
    if (x + 1 < 16) img[y * 16 + x + 1] = 0.6;
  }
  return img;
}

// Draw a 16x16 grid of cells as a square, centered in the canvas.
function drawGrid(ctx, values, width, height, min, max, color) {
  const n = 16;
  const side = Math.min(width, height) * 0.95;
  const cell = side / n;
  const x0 = (width - side) / 2;
  const y0 = (height - side) / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = color;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = values[r * n + c];
      const a = max > min ? (v - min) / (max - min) : 0;
      ctx.globalAlpha = Math.max(0, Math.min(1, a));
      ctx.fillRect(x0 + c * cell, y0 + r * cell, cell, cell);
    }
  }
  ctx.globalAlpha = 1;
}

const fmt = (x) => Number(x.toPrecision(4));

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const img = pattern16();
  const N = 256;

  const raster = ui.raster(figures, { height: 220, xLabel: 'time', xUnit: 'steps' });
  const inCanvas = ui.canvas(figures, { height: 220 });
  const countCanvas = ui.canvas(figures, { height: 220 });

  const state = { time_window: 30, ratio: 5 / 30, seed: 42, encoder: 'simple' };
  let data = null; // { spikes, ticks, T, maxCount, prefix }
  let curT = 0;
  let player = null;

  function renderInput() {
    drawGrid(inCanvas.ctx, img, inCanvas.width, inCanvas.height, 0, 1, ui.color('c1'));
  }

  // Reveal the raster and fill the count grid with the spikes up to time t.
  function renderAt(t) {
    if (!data) return;
    curT = t;
    raster.draw(data.ticks, N, data.T, { revealT: t });
    const upto = Math.max(0, Math.min(Math.floor(t) + 1, data.T));
    const counts = new Array(N);
    for (let i = 0; i < N; i++) counts[i] = data.prefix[i][upto];
    drawGrid(countCanvas.ctx, counts, countCanvas.width, countCanvas.height, 0, data.maxCount, ui.color('c2'));
  }

  function encode() {
    const T = state.time_window;
    const r = state.ratio;
    const s = state.seed;
    let spikes;
    if (state.encoder === 'i2l') spikes = M.intensity2Latency(img, { time_window: T });
    else if (state.encoder === 'poisson') spikes = M.poisson(img, { time_window: T, ratio: r, seed: s });
    else spikes = M.simplePoisson(img, { time_window: T, ratio: r, seed: s });
    const ticks = [];
    const final = new Array(N).fill(0);
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) if (spikes[t * N + i]) { ticks.push([t, i]); final[i]++; }
    // prefix[i][t] = spikes of pixel i at times < t, so counts up to t is an index
    const prefix = new Array(N);
    for (let i = 0; i < N; i++) {
      const row = new Array(T + 1).fill(0);
      for (let t = 0; t < T; t++) row[t + 1] = row[t] + (spikes[t * N + i] ? 1 : 0);
      prefix[i] = row;
    }
    data = { spikes, ticks, T, maxCount: Math.max(1, ...final), prefix };
    renderInput();
    if (player) { player.setDuration(T); renderAt(Math.min(curT, T)); }
    else renderAt(T); // no player (checker): show the whole window
  }

  const sec = ui.section(controls, 'Encoder');
  ui.slider(sec, { key: 'time_window', label: 'time_window', unit: 'steps', min: 10, max: 100, step: 1, value: 30, onInput: (v) => { state.time_window = v; encode(); } });
  ui.slider(sec, { key: 'ratio', label: 'ratio', unit: 'a.u.', min: 0.05, max: 1.0, step: 0.05, value: 5 / 30, onInput: (v) => { state.ratio = v; encode(); } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: 42, onInput: (v) => { state.seed = v; encode(); } });
  ui.select(sec, {
    key: 'encoder', label: 'encoder', value: 'simple',
    options: [
      { value: 'simple', label: 'SimplePoisson' },
      { value: 'poisson', label: 'Poisson' },
      { value: 'i2l', label: 'Intensity2Latency' },
    ],
    onInput: (v) => { state.encoder = v; encode(); },
  });

  encode();
  // the cross-area checker's mock ui has no player; fall back to the full view
  player = (typeof ui.player === 'function') ? ui.player(controls, {
    duration: state.time_window, unit: 'steps', playSeconds: 5, loop: true,
    onFrame: (t) => renderAt(t),
  }) : null;
  if (player) player.seek(0); // start empty; the player then fills the window

  // The real ui.js canvas clears itself on window resize and calls the
  // registered redraw; the cross-area checker's mock canvas has no onResize,
  // so guard it (renderAt is already called directly at mount).
  if (typeof inCanvas.onResize === 'function') inCanvas.onResize(renderInput);
  if (typeof countCanvas.onResize === 'function') countCanvas.onResize(() => renderAt(curT));
  ui.onTheme(() => { renderInput(); renderAt(curT); });

  ui.note(notes, `A grayscale image (values 0-1) is turned into a spike train. <b>SimplePoisson</b> and <b>Poisson</b> are rate codes: a pixel with intensity <var>i</var> spikes at rate <var>i</var>·ratio, so brighter pixels fire more often. <b>Intensity2Latency</b> is a latency code: brighter pixels spike <i>sooner</i>, and the first spike carries the value. In the retina, ganglion cells encode light intensity both by firing rate and by response latency. The two squares show the input image (left) and the per-pixel spike count (right) on the same 16x16 scale.`);
  ui.equation(notes, `<var>spike</var>(<var>t</var>) ~ Bernoulli(<var>i</var> · ratio) &nbsp;for SimplePoisson; &nbsp; intervals ~ Poisson(1 / (<var>i</var> · ratio)) &nbsp;for Poisson.`);
  ui.note(notes, `The ratio is dimensionless (a.u.); time is in simulation steps. ML analogy: this is like turning a continuous feature vector into a stochastic binary code (e.g. a dropout-style or stochastic binary embedding). The analogy breaks because real neurons have refractory periods and adaptation that a per-step Bernoulli does not model.`);
  ui.code(notes, () => {
    const cls = state.encoder === 'i2l' ? 'Intensity2Latency' : state.encoder === 'poisson' ? 'Poisson' : 'SimplePoisson';
    const ratioArg = state.encoder === 'i2l' ? '' : `, ratio=${fmt(state.ratio)}`;
    return `import torch
from conex.helpers.transforms.encoders import ${cls}
img = torch.rand(16, 16)  # 16x16, values in [0, 1]
spikes = ${cls}(time_window=${state.time_window}${ratioArg})(img)`;
  });
  return { destroy() { if (player) player.pause(); } };
}
