// Plate: Poisson & latency encoding of a 16x16 grayscale pattern.
// Live mode (AMENDMENT 4): the encoder window is revealed step by step.
// Hero (AMENDMENT 5): the image pixels emit a spark for every real spike, and
// the spark flies into the raster at that pixel's row.
import * as M from '../js/models/plasticity.js';
import * as A from '../js/anim.js';

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
  const FLIGHT = 4; // model steps a spark takes to fly from the pixel to the raster

  ui.tryThis(figures, 'Try this: switch encoder to Intensity2Latency and watch bright pixels spike sooner.');

  const hero = ui.hero(figures, { height: 300 });
  const raster = ui.raster(figures, { height: 220, xLabel: 'time', xUnit: 'steps' });
  const inCanvas = ui.canvas(figures, { height: 220 });
  const countCanvas = ui.canvas(figures, { height: 220 });

  const caption = (c, text) => {
    if (!c.canvas.parentElement) return;
    const cap = document.createElement('figcaption');
    cap.textContent = text;
    c.canvas.parentElement.appendChild(cap);
  };
  caption(inCanvas, '1. Input intensity (0-1)');
  caption(countCanvas, '2. Accumulated spike count (last window)');

  const state = { time_window: 30, ratio: 5 / 30, seed: 42, encoder: 'simple' };
  let data = null; // { spikes, T, maxCount }
  let t = 0;
  const sparks = []; // {i, t0}

  function encode() {
    const T = state.time_window;
    const r = state.ratio;
    const s = state.seed;
    let spikes;
    if (state.encoder === 'i2l') spikes = M.intensity2Latency(img, { time_window: T });
    else if (state.encoder === 'poisson') spikes = M.poisson(img, { time_window: T, ratio: r, seed: s });
    else spikes = M.simplePoisson(img, { time_window: T, ratio: r, seed: s });
    let maxCount = 1;
    for (let i = 0; i < N; i++) {
      let c = 0;
      for (let tt = 0; tt < T; tt++) if (spikes[tt * N + i]) c++;
      maxCount = Math.max(maxCount, c);
    }
    data = { spikes, T, maxCount };
  }

  // The window is treated as periodic so the live reveal scrolls without a jump.
  function spikeAt(tt, i) {
    const s = ((tt % data.T) + data.T) % data.T;
    return data.spikes[s * N + i];
  }

  function step() {
    const s = t % data.T;
    for (let i = 0; i < N; i++) if (spikeAt(t, i)) sparks.push({ i, t0: t });
    t += 1;
    while (sparks.length && t - sparks[0].t0 >= FLIGHT) sparks.shift();
  }

  function reset() {
    t = 0; sparks.length = 0;
  }

  function renderInput() {
    drawGrid(inCanvas.ctx, img, inCanvas.width, inCanvas.height, 0, 1, ui.color('c1'));
  }

  function drawHero() {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const T = data.T;

    // image grid (left)
    const cell = Math.min(W * 0.34, H * 0.9) / 16;
    const gx0 = W * 0.04, gy0 = (H - 16 * cell) / 2;
    ctx.fillStyle = c1;
    for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
      const v = img[r * 16 + c];
      if (v <= 0) continue;
      ctx.globalAlpha = v;
      ctx.fillRect(gx0 + c * cell, gy0 + r * cell, cell, cell);
    }
    ctx.globalAlpha = 1;

    // raster strip (right): x = time over the last T steps, y = pixel row
    const rx0 = W * 0.5, rx1 = W * 0.96;
    const ry = (i) => H * 0.06 + (i / (N - 1)) * (H * 0.88);
    const rx = (tt) => rx0 + ((tt - (t - T)) / T) * (rx1 - rx0);
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(rx0, H * 0.06); ctx.lineTo(rx0, H * 0.94); ctx.stroke();
    // dots for the spikes in the last T steps
    ctx.fillStyle = c2;
    for (let tt = Math.max(0, t - T); tt < t; tt++) {
      const x = rx(tt);
      for (let i = 0; i < N; i++) if (spikeAt(tt, i)) {
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x - 1, ry(i) - 1, 2, 2);
      }
    }
    ctx.globalAlpha = 1;
    // sparks flying from the grid cell to the raster dot
    for (const sp of sparks) {
      const p = A.encSparkProgress((t - sp.t0) / FLIGHT);
      const i = sp.i;
      const cx = gx0 + ((i % 16) + 0.5) * cell;
      const cy = gy0 + (Math.floor(i / 16) + 0.5) * cell;
      const tx = rx(t - 1), ty = ry(i);
      const x = A.lerp(cx, tx, p);
      const y = A.lerp(cy, ty, p);
      ctx.save(); ctx.shadowColor = c2; ctx.shadowBlur = 8; ctx.fillStyle = c2;
      ctx.globalAlpha = 1 - p * 0.5;
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('image', gx0 + 8 * cell, H * 0.97);
    ctx.fillText('raster (spikes)', (rx0 + rx1) / 2, H * 0.97);
  }

  function drawCount() {
    const counts = new Array(N).fill(0);
    for (let tt = Math.max(0, t - data.T); tt < t; tt++) {
      for (let i = 0; i < N; i++) if (spikeAt(tt, i)) counts[i]++;
    }
    drawGrid(countCanvas.ctx, counts, countCanvas.width, countCanvas.height, 0, data.maxCount, ui.color('c2'));
    // numeric colour bar for the count (0 .. maxCount)
    const ctx = countCanvas.ctx, Wc = countCanvas.width, Hc = countCanvas.height;
    const bx = Wc - 18, by = Hc * 0.2, bh = Hc * 0.6, bw = 8;
    const c2 = ui.color('c2');
    const grad = typeof ctx.createLinearGradient === 'function' ? ctx.createLinearGradient(0, by + bh, 0, by) : null;
    if (grad && typeof grad.addColorStop === 'function') {
      grad.addColorStop(0, ui.color('paper')); grad.addColorStop(1, c2); ctx.fillStyle = grad;
    } else ctx.fillStyle = c2;
    ctx.globalAlpha = 1;
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = ui.color('ink'); ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw, bh);
    ctx.fillStyle = ui.color('muted'); ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(String(data.maxCount), bx + bw + 2, by);
    ctx.fillText('0', bx + bw + 2, by + bh);
  }

  function onFrame() {
    drawHero();
    drawCount();
    const ticks = [];
    for (let tt = Math.max(0, t - data.T); tt < t; tt++) {
      for (let i = 0; i < N; i++) if (spikeAt(tt, i)) ticks.push([tt, i]);
    }
    raster.draw(ticks, N, data.T, { xRange: [t - data.T, t] });
    hero.narration(A.encNarration(state.encoder));
  }

  encode();
  renderInput();
  // fill the window once so the figures are not empty before the first frame
  for (let i = 0; i < data.T; i++) step();
  onFrame();

  const sec = ui.section(controls, 'Encoder');
  ui.slider(sec, { key: 'time_window', label: 'time_window', unit: 'steps', min: 10, max: 100, step: 1, value: 30, onInput: (v) => { state.time_window = v; encode(); reset(); liveHandle && liveHandle.reset(); } });
  ui.slider(sec, { key: 'ratio', label: 'ratio', unit: 'a.u.', min: 0.05, max: 1.0, step: 0.05, value: 5 / 30, onInput: (v) => { state.ratio = v; encode(); reset(); liveHandle && liveHandle.reset(); } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: 42, onInput: (v) => { state.seed = v; encode(); reset(); liveHandle && liveHandle.reset(); } });
  ui.select(sec, {
    key: 'encoder', label: 'encoder', value: 'simple',
    options: [
      { value: 'simple', label: 'SimplePoisson' },
      { value: 'poisson', label: 'Poisson' },
      { value: 'i2l', label: 'Intensity2Latency' },
    ],
    onInput: (v) => { state.encoder = v; encode(); reset(); liveHandle && liveHandle.reset(); },
  });

  let liveHandle = null;
  liveHandle = ui.live(controls, {
    dt: 1, unit: 'steps', modelPerSecond: 12, windowSize: data.T,
    step, reset, onFrame,
  });

  if (typeof inCanvas.onResize === 'function') inCanvas.onResize(renderInput);
  if (typeof countCanvas.onResize === 'function') countCanvas.onResize(() => { drawCount(); });
  hero.onResize(onFrame);
  ui.onTheme(() => { renderInput(); onFrame(); });

  ui.note(notes, `A grayscale image (values 0-1) is turned into a spike train. <b>SimplePoisson</b> and <b>Poisson</b> are rate codes: a pixel with intensity <var>i</var> spikes at rate <var>i</var>·ratio, so brighter pixels fire more often. <b>Intensity2Latency</b> is a latency code: brighter pixels spike <i>sooner</i>, and the first spike carries the value. In the retina, ganglion cells encode light intensity both by firing rate and by response latency. The hero shows the image on the left; every real spike emits a spark that flies into the raster at that pixel's row.`);
  ui.equation(notes, `<var>spike</var>(<var>t</var>) ~ Bernoulli(<var>i</var> · ratio) &nbsp;for SimplePoisson; &nbsp; intervals ~ Poisson(1 / (<var>i</var> · ratio)) &nbsp;for Poisson.`);
  ui.note(notes, `The ratio is dimensionless (a.u.); time is in simulation steps. Note the <b>Poisson</b> encoder (the BindsNET method) samples the inter-spike <i>intervals</i> from a discrete Poisson distribution with mean 1/(<var>i</var>·ratio) steps (a zero interval becomes 1), then cumsums them to spike times. This is not the textbook Poisson process, whose intervals are exponential (continuous) or geometric (discrete); it is a BindsNET/CoNeX heuristic. ML analogy: this is like turning a continuous feature vector into a stochastic binary code. The analogy breaks because real neurons have refractory periods and adaptation that a per-step Bernoulli does not model.`);
  ui.code(notes, () => {
    const cls = state.encoder === 'i2l' ? 'Intensity2Latency' : state.encoder === 'poisson' ? 'Poisson' : 'SimplePoisson';
    const ratioArg = state.encoder === 'i2l' ? '' : `, ratio=${fmt(state.ratio)}`;
    return `import torch
from conex.helpers.transforms.encoders import ${cls}
img = torch.rand(16, 16)  # 16x16, values in [0, 1]
spikes = ${cls}(time_window=${state.time_window}${ratioArg})(img)`;
  });
  return { destroy() { if (liveHandle) liveHandle.pause(); } };
}
