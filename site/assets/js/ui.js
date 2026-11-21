// Shared UI + plotting API for the CoNeX site. See .worker/CONTRACT.md.
import { encodeHash, decodeHash, rng, computeTicks, dataRange, fmtTick, formatValue, listenerList, clampTime, playerAdvance, playerInteger, playerShownTime, playerSpeedFromPos, playerPosFromSpeed, liveSteps, LIVE_MAX_STEPS, ease, readoutText } from './core.js';

const THEME_KEY = 'conex-theme';
const SPEED_KEY = 'conex-speed';

// The last chosen playback speed, shared by every player. Falls back to 1x.
function readSpeed() {
  try {
    const v = Number(localStorage.getItem(SPEED_KEY));
    if (Number.isFinite(v) && v >= 0.1 && v <= 10) return v;
  } catch {
    // storage unavailable; use the default
  }
  return 1;
}
function writeSpeed(v) {
  try {
    localStorage.setItem(SPEED_KEY, String(v));
  } catch {
    // ignore
  }
}
function fmtSpeed(v) {
  return `${v.toFixed(1)}x`;
}

const PALETTES = {
  light: {
    paper: '#fbfaf5', ink: '#1f2a2c', muted: '#5f6b6d', rule: '#dcdccd',
    c1: '#0f7b7f', c2: '#d95b43', c3: '#b98a1f', c4: '#3d4f9f', c5: '#5b8a3c', c6: '#8b3f73',
  },
  dark: {
    paper: '#141b1c', ink: '#e5ebe7', muted: '#9aa7a6', rule: '#2c3738',
    c1: '#3cb4b9', c2: '#ef7d66', c3: '#d9ab3f', c4: '#8294dd', c5: '#86b468', c6: '#c26fa9',
  },
};

let theme = 'light';
try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'light' || saved === 'dark') theme = saved;
} catch {
  // storage unavailable (e.g. file://); stay light
}
const themeListeners = [];

function applyTheme() {
  document.documentElement.setAttribute('data-theme', theme);
}
applyTheme();

export function setTheme(t) {
  if (t !== 'light' && t !== 'dark') return;
  theme = t;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // ignore
  }
  applyTheme();
  for (const fn of themeListeners) fn(theme);
}

export function getTheme() {
  return theme;
}

// --- URL-hash state -------------------------------------------------------

function readHash() {
  return decodeHash(typeof location !== 'undefined' ? location.hash : '');
}

function writeHash(params) {
  const h = readHash();
  Object.assign(h, params);
  history.replaceState(null, '', '#' + encodeHash(h));
}

// Validate a raw hash string against a control's spec; return null if invalid.
function fromHash(raw, spec) {
  if (raw === undefined) return null;
  if (spec.type === 'number') {
    const v = Number(raw);
    if (!Number.isFinite(v)) return null;
    if (spec.min !== undefined && v < spec.min) return null;
    if (spec.max !== undefined && v > spec.max) return null;
    return v;
  }
  if (spec.type === 'select') {
    return spec.options.some((o) => o.value === raw) ? raw : null;
  }
  if (spec.type === 'toggle') {
    if (raw === '1' || raw === 'true') return true;
    if (raw === '0' || raw === 'false') return false;
    return null;
  }
  return null;
}

function toHash(v, spec) {
  if (spec.type === 'toggle') return v ? '1' : '0';
  return String(v);
}

// All keyed controls, so plate.html can download them as JSON.
const keyedControls = new Map();

export function keyedParams() {
  const out = {};
  for (const [key, c] of keyedControls) out[key] = c.get();
  return out;
}

function registerControl(key, spec, control) {
  keyedControls.set(key, control);
  const raw = readHash()[key];
  const v = fromHash(raw, spec);
  if (v !== null) control.set(v, { silent: true });
}

function commitControl(key, spec, control) {
  writeHash({ [key]: toHash(control.get(), spec) });
}

// --- Layout ----------------------------------------------------------------

export function layout(root) {
  const wrap = document.createElement('div');
  wrap.className = 'plate-layout';
  const controls = document.createElement('div');
  controls.className = 'plate-controls';
  const figures = document.createElement('div');
  figures.className = 'plate-figures';
  const notes = document.createElement('div');
  notes.className = 'plate-notes';
  wrap.append(controls, figures, notes);
  root.appendChild(wrap);
  return { controls, figures, notes };
}

export function section(parent, title) {
  const sec = document.createElement('section');
  sec.className = 'ui-section';
  const h = document.createElement('h3');
  h.textContent = title;
  sec.appendChild(h);
  parent.appendChild(sec);
  return sec;
}

// --- Controls --------------------------------------------------------------

export function slider(parent, { key, label, unit, min, max, step, value, onInput }) {
  const row = document.createElement('label');
  row.className = 'ui-control';
  const head = document.createElement('span');
  head.className = 'ui-control-head';
  const name = document.createElement('span');
  name.textContent = label;
  const readout = document.createElement('span');
  readout.className = 'ui-readout';
  head.append(name, readout);
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  row.append(head, input);
  parent.appendChild(row);

  const spec = { type: 'number', min, max };
  const listeners = listenerList();
  const control = {
    get() {
      return Number(input.value);
    },
    set(v) {
      input.value = String(v);
      readout.textContent = `${formatValue(v, step)} ${unit}`;
    },
    onInput(fn) {
      return listeners.add(fn);
    },
  };
  control.set(value);
  input.addEventListener('input', () => {
    const v = Number(input.value);
    readout.textContent = `${formatValue(v, step)} ${unit}`;
    if (key) commitControl(key, spec, control);
    if (onInput) onInput(v);
    listeners.emit(v);
  });
  if (key) registerControl(key, spec, control);
  return control;
}

export function number(parent, { key, label, unit, value, onInput }) {
  const row = document.createElement('label');
  row.className = 'ui-control ui-control-number';
  const name = document.createElement('span');
  name.textContent = label;
  const input = document.createElement('input');
  input.type = 'number';
  input.value = String(value);
  const u = document.createElement('span');
  u.className = 'ui-unit';
  u.textContent = unit;
  row.append(name, input, u);
  parent.appendChild(row);

  const spec = { type: 'number' };
  const listeners = listenerList();
  const control = {
    get() {
      return Number(input.value);
    },
    set(v) {
      input.value = String(v);
    },
    onInput(fn) {
      return listeners.add(fn);
    },
  };
  input.addEventListener('input', () => {
    const v = Number(input.value);
    if (key) commitControl(key, spec, control);
    if (onInput) onInput(v);
    listeners.emit(v);
  });
  if (key) registerControl(key, spec, control);
  return control;
}

export function select(parent, { key, label, options, value, onInput }) {
  const row = document.createElement('label');
  row.className = 'ui-control';
  const name = document.createElement('span');
  name.textContent = label;
  const input = document.createElement('select');
  for (const o of options) {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    input.appendChild(opt);
  }
  input.value = value;
  row.append(name, input);
  parent.appendChild(row);

  const spec = { type: 'select', options };
  const listeners = listenerList();
  const control = {
    get() {
      return input.value;
    },
    set(v) {
      input.value = v;
    },
    onInput(fn) {
      return listeners.add(fn);
    },
  };
  input.addEventListener('change', () => {
    if (key) commitControl(key, spec, control);
    if (onInput) onInput(input.value);
    listeners.emit(input.value);
  });
  if (key) registerControl(key, spec, control);
  return control;
}

export function toggle(parent, { key, label, value, onInput }) {
  const row = document.createElement('label');
  row.className = 'ui-control ui-control-toggle';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = !!value;
  const name = document.createElement('span');
  name.textContent = label;
  row.append(input, name);
  parent.appendChild(row);

  const spec = { type: 'toggle' };
  const listeners = listenerList();
  const control = {
    get() {
      return input.checked;
    },
    set(v) {
      input.checked = !!v;
    },
    onInput(fn) {
      return listeners.add(fn);
    },
  };
  input.addEventListener('change', () => {
    if (key) commitControl(key, spec, control);
    if (onInput) onInput(input.checked);
    listeners.emit(input.checked);
  });
  if (key) registerControl(key, spec, control);
  return control;
}

export function button(parent, { label, onClick }) {
  const b = document.createElement('button');
  b.className = 'ui-button';
  b.textContent = label;
  b.addEventListener('click', onClick);
  parent.appendChild(b);
  return b;
}

// --- Figures ---------------------------------------------------------------

// HiDPI canvas: CSS size from the parent, backing store scaled by devicePixelRatio.
function makeCanvas(parent, height) {
  const fig = document.createElement('figure');
  fig.className = 'ui-figure';
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = `${height}px`;
  fig.appendChild(canvas);
  parent.appendChild(fig);
  const state = { canvas, fig, height, width: 0, ctx: canvas.getContext('2d') };

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    state.width = canvas.clientWidth || 300;
    canvas.width = Math.round(state.width * dpr);
    canvas.height = Math.round(height * dpr);
    state.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  const ro = new ResizeObserver(() => {
    resize();
    if (state.redraw) state.redraw();
  });
  ro.observe(canvas);
  state._ro = ro;
  state._resize = resize;
  return state;
}

// "label (unit)" when a unit is given, just "label" when it is empty.
function axisTitle(label, unit) {
  return unit ? `${label} (${unit})` : label;
}

function drawAxes(ctx, w, h, m, xRange, yRange, xTicks, yTicks, xLabel, yLabel, xUnit, yUnit) {
  const pal = PALETTES[theme];
  const x0 = m.left, x1 = w - m.right, y0 = h - m.bottom, y1 = m.top;
  const sx = (x) => x0 + ((x - xRange[0]) / (xRange[1] - xRange[0])) * (x1 - x0);
  const sy = (y) => y0 - ((y - yRange[0]) / (yRange[1] - yRange[0])) * (y0 - y1);

  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = pal.muted;
  ctx.strokeStyle = pal.rule;
  ctx.lineWidth = 1;

  for (const t of xTicks.ticks) {
    const px = sx(t);
    ctx.beginPath();
    ctx.moveTo(px, y1);
    ctx.lineTo(px, y0);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(fmtTick(t, xTicks.step), px, y0 + 4);
  }
  for (const t of yTicks.ticks) {
    const py = sy(t);
    ctx.beginPath();
    ctx.moveTo(x0, py);
    ctx.lineTo(x1, py);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmtTick(t, yTicks.step), x0 - 5, py);
  }
  // frame
  ctx.strokeStyle = pal.ink;
  ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);

  ctx.fillStyle = pal.ink;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(axisTitle(xLabel, xUnit), (x0 + x1) / 2, h - 2);
  ctx.save();
  ctx.translate(10, (y0 + y1) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(axisTitle(yLabel, yUnit), 0, 0);
  ctx.restore();
  return { sx, sy, x0, x1, y0, y1 };
}

export function plot(parent, { height = 260, xLabel = 'x', yLabel = 'y', xUnit = 'a.u.', yUnit = 'a.u.' }) {
  const st = makeCanvas(parent, height);
  const M = { left: 48, right: 12, top: 12, bottom: 34 };
  let last = null;

  function render() {
    if (!last) return;
    const { series, opts } = last;
    const w = st.width, h = st.height;
    const ctx = st.ctx;
    const pal = PALETTES[theme];
    ctx.clearRect(0, 0, w, h);

    let xRange = opts.xRange, yRange = opts.yRange;
    if (!xRange || !yRange) {
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const s of series) {
        for (const x of s.x) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; }
        for (const y of s.y) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
      }
      if (!Number.isFinite(xmin)) { xmin = 0; xmax = 1; ymin = 0; ymax = 1; }
      if (!xRange) xRange = dataRange(xmin, xmax);
      if (!yRange) yRange = dataRange(ymin, ymax);
    }
    const xTicks = computeTicks(xRange[0], xRange[1]);
    const yTicks = computeTicks(yRange[0], yRange[1]);
    const ax = drawAxes(ctx, w, h, M, xRange, yRange, xTicks, yTicks, xLabel, yLabel, xUnit, yUnit);

    ctx.save();
    ctx.beginPath();
    ctx.rect(ax.x0, ax.y1, ax.x1 - ax.x0, ax.y0 - ax.y1);
    ctx.clip();

    const revealX = opts.revealX;
    for (const s of series) {
      const color = ui.color(s.color || 'c1');
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      // when revealing, draw only the prefix with x <= revealX (time series are sorted)
      let lastIdx = s.x.length - 1;
      if (typeof revealX === 'number') {
        lastIdx = -1;
        for (let i = 0; i < s.x.length; i++) {
          if (s.x[i] <= revealX) lastIdx = i;
          else break;
        }
      }
      if (lastIdx < 0) continue;
      if (s.style === 'dots') {
        for (let i = 0; i <= lastIdx; i++) {
          ctx.beginPath();
          ctx.arc(ax.sx(s.x[i]), ax.sy(s.y[i]), 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      } else if (s.style === 'bars') {
        const bw = Math.max(1, (ax.x1 - ax.x0) / s.x.length * 0.8);
        for (let i = 0; i <= lastIdx; i++) {
          const px = ax.sx(s.x[i]);
          const py = ax.sy(s.y[i]);
          ctx.fillRect(px - bw / 2, Math.min(py, ax.y0), bw, Math.abs(ax.y0 - py));
        }
      } else {
        ctx.beginPath();
        for (let i = 0; i <= lastIdx; i++) {
          const px = ax.sx(s.x[i]);
          const py = ax.sy(s.y[i]);
          if (i === 0) ctx.moveTo(px, py);
          else if (s.style === 'step') {
            ctx.lineTo(px, ax.sy(s.y[i - 1]));
            ctx.lineTo(px, py);
          } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      if (typeof revealX === 'number') {
        ctx.beginPath();
        ctx.arc(ax.sx(s.x[lastIdx]), ax.sy(s.y[lastIdx]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (typeof revealX === 'number') {
      const px = ax.sx(revealX);
      ctx.strokeStyle = pal.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, ax.y1);
      ctx.lineTo(px, ax.y0);
      ctx.stroke();
    }

    for (const hl of opts.hlines || []) {
      const py = ax.sy(hl.y);
      ctx.strokeStyle = pal.muted;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(ax.x0, py);
      ctx.lineTo(ax.x1, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(hl.label, ax.x0 + 4, py - 2);
    }
    for (const vl of opts.vlines || []) {
      const px = ax.sx(vl.x);
      ctx.strokeStyle = pal.muted;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, ax.y1);
      ctx.lineTo(px, ax.y0);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(vl.label, px + 3, ax.y1 + 2);
    }
    ctx.restore();

    // legend
    const labeled = series.filter((s) => s.label);
    if (labeled.length) {
      ctx.font = '11px system-ui, sans-serif';
      let lx = ax.x0 + 6;
      const ly = ax.y1 + 12;
      for (const s of labeled) {
        ctx.strokeStyle = ui.color(s.color || 'c1');
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(lx + 14, ly);
        ctx.stroke();
        ctx.fillStyle = pal.ink;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(s.label, lx + 18, ly);
        lx += 18 + ctx.measureText(s.label).width + 12;
      }
    }
  }

  st.redraw = render;
  ui.onTheme(render);

  return {
    draw(series, opts = {}) {
      last = { series, opts };
      render();
    },
  };
}

export function raster(parent, { height = 200, xLabel = 'time', xUnit = 'ms' }) {
  const st = makeCanvas(parent, height);
  const M = { left: 48, right: 12, top: 8, bottom: 34 };
  let last = null;

  function render() {
    if (!last) return;
    const { spikes, nRows, tMax, revealT, xRange } = last;
    const w = st.width, h = st.height;
    const ctx = st.ctx;
    const pal = PALETTES[theme];
    ctx.clearRect(0, 0, w, h);
    // xRange: [a, b] scrolls the axis (live mode); otherwise [0, tMax]
    const xr = Array.isArray(xRange) ? xRange : [0, tMax || 1];
    const xTicks = computeTicks(xr[0], xr[1]);
    const yTicks = { ticks: [0, nRows], step: nRows };
    const ax = drawAxes(ctx, w, h, M, xr, [0, nRows], xTicks, yTicks, xLabel, 'neuron', xUnit, 'index');
    ctx.fillStyle = pal.ink;
    for (const [t, idx] of spikes) {
      if (idx < 0 || idx >= nRows) continue;
      if (xRange && (t < xr[0] || t > xr[1])) continue;
      if (typeof revealT === 'number' && t > revealT) continue;
      const px = ax.sx(t);
      const py = ax.y1 + ((idx + 0.5) / nRows) * (ax.y0 - ax.y1);
      ctx.fillRect(px - 0.75, py - 3, 1.5, 6);
    }
    if (typeof revealT === 'number') {
      const px = ax.sx(revealT);
      ctx.strokeStyle = pal.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, ax.y1);
      ctx.lineTo(px, ax.y0);
      ctx.stroke();
    }
  }

  st.redraw = render;
  ui.onTheme(render);

  return {
    draw(spikes, nRows, tMax, { revealT, xRange } = {}) {
      last = { spikes, nRows, tMax, revealT, xRange };
      render();
    },
  };
}

export function heatmap(parent, { rows, cols, label = '', height = 200 }) {
  const st = makeCanvas(parent, height);
  const M = { left: 48, right: 12, top: 8, bottom: 34 };
  let last = null;

  function render() {
    if (!last) return;
    const { values, min, max } = last;
    const w = st.width, h = st.height;
    const ctx = st.ctx;
    const pal = PALETTES[theme];
    ctx.clearRect(0, 0, w, h);
    let lo = min, hi = max;
    if (lo === undefined || hi === undefined) {
      lo = Infinity; hi = -Infinity;
      for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (lo === hi) hi = lo + 1;
    }
    const cw = (w - M.left - M.right) / cols;
    const ch = (h - M.top - M.bottom) / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r * cols + c];
        const f = (v - lo) / (hi - lo);
        ctx.fillStyle = mix(pal.paper, pal.c1, Math.max(0, Math.min(1, f)));
        ctx.fillRect(M.left + c * cw, M.top + r * ch, cw + 0.5, ch + 0.5);
      }
    }
    ctx.strokeStyle = pal.ink;
    ctx.strokeRect(M.left, M.top, cols * cw, rows * ch);
    ctx.fillStyle = pal.ink;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, (M.left + w - M.right) / 2, h - 2);
  }

  // redraw on resize (the RO resets the canvas) and on theme change
  st.redraw = render;
  ui.onTheme(render);

  return {
    draw(values, { min, max } = {}) {
      last = { values, min, max };
      render();
    },
  };
}

// Blend two hex colors; t in [0,1].
function mix(a, b, t) {
  const pa = hex(a), pb = hex(b);
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function hex(s) {
  const m = s.match(/^#([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function canvas(parent, { height = 200 }) {
  const st = makeCanvas(parent, height);
  const listeners = listenerList();
  // the plate draws itself; the RO resets the canvas on resize, so tell it to redraw
  st.redraw = () => listeners.emit();
  return {
    get canvas() { return st.canvas; },
    get ctx() { return st.ctx; },
    get width() { return st.width; },
    get height() { return st.height; },
    onResize(fn) {
      return listeners.add(fn);
    },
  };
}

// --- Text ------------------------------------------------------------------

export function note(parent, html) {
  const p = document.createElement('p');
  p.className = 'ui-note';
  p.innerHTML = html;
  parent.appendChild(p);
  return p;
}

export function equation(parent, html) {
  const div = document.createElement('div');
  div.className = 'ui-equation';
  div.innerHTML = html;
  parent.insertBefore(div, parent.firstChild);
  return div;
}

export function code(parent, fn) {
  const box = document.createElement('div');
  box.className = 'ui-codebox';
  const bar = document.createElement('div');
  bar.className = 'ui-codebar';
  const title = document.createElement('span');
  title.textContent = 'CoNeX code';
  const copy = document.createElement('button');
  copy.className = 'ui-button ui-button-small';
  copy.textContent = 'Copy';
  bar.append(title, copy);
  const pre = document.createElement('pre');
  const codeEl = document.createElement('code');
  pre.appendChild(codeEl);
  box.append(bar, pre);
  parent.appendChild(box);

  function refresh() {
    codeEl.textContent = fn();
  }
  refresh();
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(codeEl.textContent);
      copy.textContent = 'Copied';
    } catch {
      copy.textContent = 'Select and copy';
    }
    setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
  });
  return { refresh };
}

// --- Misc ------------------------------------------------------------------

export { rng };

export function color(name) {
  return PALETTES[theme][name] || PALETTES[theme].ink;
}

export function onTheme(fn) {
  themeListeners.push(fn);
  return () => {
    const i = themeListeners.indexOf(fn);
    if (i >= 0) themeListeners.splice(i, 1);
  };
}

export function animate(fn) {
  let raf = 0;
  let lastT = 0;
  let running = false;
  function loop(t) {
    if (!running) return;
    const dt = lastT ? t - lastT : 16;
    lastT = t;
    fn(dt);
    raf = requestAnimationFrame(loop);
  }
  return {
    start() {
      if (running) return;
      running = true;
      lastT = 0;
      raf = requestAnimationFrame(loop);
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
    },
    running() {
      return running;
    },
  };
}

// A time player: reveals a precomputed run over model time. Model time advances
// at speed * duration / playSeconds per real second. See AMENDMENT 2.
export function player(parent, { duration, unit = 'ms', playSeconds = 5, loop = true, integer, onFrame, readoutFn }) {
  let dur = duration;
  let t = 0;
  let speed = readSpeed();
  let playing = false;
  let raf = 0;
  let lastTs = 0;
  const isInt = playerInteger(unit, integer);

  const wrap = document.createElement('div');
  wrap.className = 'ui-player';

  const row = document.createElement('div');
  row.className = 'ui-player-row';

  const playBtn = document.createElement('button');
  playBtn.className = 'ui-button ui-button-small';
  playBtn.type = 'button';
  playBtn.textContent = 'Play';

  const restartBtn = document.createElement('button');
  restartBtn.className = 'ui-button ui-button-small';
  restartBtn.type = 'button';
  restartBtn.textContent = 'Restart';

  const speedWrap = document.createElement('label');
  speedWrap.className = 'ui-player-speed';
  const speedName = document.createElement('span');
  speedName.textContent = 'speed';
  const speedReadout = document.createElement('span');
  speedReadout.className = 'ui-readout';
  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '0';
  speedSlider.max = '1';
  speedSlider.step = '0.01';
  speedSlider.value = String(playerPosFromSpeed(speed));
  speedSlider.setAttribute('aria-label', 'Playback speed');
  speedReadout.textContent = fmtSpeed(speed);
  speedWrap.append(speedName, speedReadout, speedSlider);

  const readout = document.createElement('span');
  readout.className = 'ui-readout ui-player-readout';

  row.append(playBtn, restartBtn, speedWrap, readout);

  const scrub = document.createElement('input');
  scrub.type = 'range';
  scrub.className = 'ui-player-scrub';
  scrub.min = '0';
  scrub.max = String(dur);
  scrub.step = String(dur / 1000);
  scrub.value = '0';

  wrap.append(row, scrub);
  parent.appendChild(wrap);

  function fmtTime(v) {
    return readoutText(v, { unit, isInt, decimals: dur < 10 ? 2 : 1, readout: readoutFn });
  }

  function update() {
    scrub.value = String(t);
    readout.textContent = fmtTime(playerShownTime(t, isInt));
  }

  function frame(ts) {
    if (!playing) return;
    const dtMs = lastTs ? ts - lastTs : 0;
    lastTs = ts;
    if (dtMs > 0) {
      t = playerAdvance(t, dtMs, dur, playSeconds, speed, loop);
      update();
      if (onFrame) onFrame(playerShownTime(t, isInt));
    }
    if (!loop && t >= dur) {
      pause();
      return;
    }
    raf = requestAnimationFrame(frame);
  }

  function play() {
    if (playing) return;
    if (!loop && t >= dur) t = 0; // replay from the start
    playing = true;
    lastTs = 0;
    playBtn.textContent = 'Pause';
    raf = requestAnimationFrame(frame);
  }

  function pause() {
    playing = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(raf);
  }

  function seek(nt) {
    t = clampTime(nt, dur);
    update();
    if (onFrame) onFrame(playerShownTime(t, isInt));
  }

  function setDuration(d) {
    dur = d;
    scrub.max = String(dur);
    scrub.step = String(dur / 1000);
    t = clampTime(t, dur);
    update();
    if (onFrame) onFrame(playerShownTime(t, isInt));
  }

  playBtn.addEventListener('click', () => (playing ? pause() : play()));
  restartBtn.addEventListener('click', () => {
    seek(0);
    play();
  });
  speedSlider.addEventListener('input', () => {
    speed = playerSpeedFromPos(Number(speedSlider.value));
    speedReadout.textContent = fmtSpeed(speed);
    writeSpeed(speed);
  });
  scrub.addEventListener('input', () => {
    seek(Number(scrub.value));
  });

  // pause while the tab is hidden so time does not jump on return
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause();
  });

  const reduced =
    typeof matchMedia === 'function' &&
    matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) {
    t = dur;
    update();
    if (onFrame) onFrame(playerShownTime(t, isInt));
  } else {
    play();
  }

  return { play, pause, seek, playing: () => playing, setDuration };
}

// A live model runner: steps the plate's own model each frame, like a live
// oscilloscope. Model time advances at speed * modelPerSecond per real
// second; step() is called once per dt of model time. See AMENDMENT 4.
export function live(parent, { dt, unit = 'ms', modelPerSecond, windowSize, step, reset, onFrame, readoutFn }) {
  let t = 0;
  let acc = 0;
  let speed = readSpeed();
  let playing = false;
  let raf = 0;
  let lastTs = 0;
  let slowed = false;

  const wrap = document.createElement('div');
  wrap.className = 'ui-live';

  const row = document.createElement('div');
  row.className = 'ui-player-row';

  const playBtn = document.createElement('button');
  playBtn.className = 'ui-button ui-button-small';
  playBtn.type = 'button';
  playBtn.textContent = 'Play';

  const resetBtn = document.createElement('button');
  resetBtn.className = 'ui-button ui-button-small';
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset';

  const speedWrap = document.createElement('label');
  speedWrap.className = 'ui-player-speed';
  const speedName = document.createElement('span');
  speedName.textContent = 'speed';
  const speedReadout = document.createElement('span');
  speedReadout.className = 'ui-readout';
  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = '0';
  speedSlider.max = '1';
  speedSlider.step = '0.01';
  speedSlider.value = String(playerPosFromSpeed(speed));
  speedSlider.setAttribute('aria-label', 'Playback speed');
  speedReadout.textContent = fmtSpeed(speed);
  speedWrap.append(speedName, speedReadout, speedSlider);

  const readout = document.createElement('span');
  readout.className = 'ui-readout ui-player-readout';

  row.append(playBtn, resetBtn, speedWrap, readout);
  wrap.append(row);
  parent.appendChild(wrap);

  function fmtTime(v) {
    return readoutText(v, { unit, decimals: t < 10 ? 2 : 1, slowed, readout: readoutFn });
  }

  function update() {
    readout.textContent = fmtTime(t);
  }

  function frame(ts) {
    if (!playing) return;
    const dtMs = lastTs ? ts - lastTs : 0;
    lastTs = ts;
    if (dtMs > 0) {
      const r = liveSteps(acc, dtMs, speed * modelPerSecond, dt);
      acc = r.acc;
      slowed = r.steps > LIVE_MAX_STEPS;
      const n = Math.min(r.steps, LIVE_MAX_STEPS);
      for (let i = 0; i < n; i++) {
        t += dt;
        step();
      }
      update();
      if (onFrame) onFrame(t);
    }
    raf = requestAnimationFrame(frame);
  }

  function play() {
    if (playing) return;
    playing = true;
    lastTs = 0;
    playBtn.textContent = 'Pause';
    raf = requestAnimationFrame(frame);
  }

  function pause() {
    playing = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(raf);
  }

  function resetAll() {
    t = 0;
    acc = 0;
    slowed = false;
    if (reset) reset();
    update();
    if (onFrame) onFrame(t);
  }

  playBtn.addEventListener('click', () => (playing ? pause() : play()));
  resetBtn.addEventListener('click', resetAll);
  speedSlider.addEventListener('input', () => {
    speed = playerSpeedFromPos(Number(speedSlider.value));
    speedReadout.textContent = fmtSpeed(speed);
    writeSpeed(speed);
  });

  // pause while the tab is hidden so time does not jump on return
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause();
  });

  const reduced =
    typeof matchMedia === 'function' &&
    matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduced) play();

  return { play, pause, playing: () => playing, reset: resetAll, t: () => t };
}

// AMENDMENT 5: the large hero canvas at the top of the figures column.
// HiDPI like ui.canvas, and the plate is told to redraw on resize AND on
// theme change (so it can pick up ui.color() again). A one-line narration
// caption sits under the canvas.
export function hero(parent, { height = 320 } = {}) {
  const st = makeCanvas(parent, height);
  const listeners = listenerList();
  st.redraw = () => listeners.emit();
  ui.onTheme(() => listeners.emit());
  const cap = document.createElement('figcaption');
  cap.className = 'ui-narration';
  st.fig.appendChild(cap);
  return {
    get canvas() { return st.canvas; },
    get ctx() { return st.ctx; },
    get width() { return st.width; },
    get height() { return st.height; },
    onResize(fn) {
      return listeners.add(fn);
    },
    narration(text) {
      cap.textContent = text;
    },
  };
}

// A short "Try this" experiment line: one sentence naming what to change and
// what to watch. Inserted at the top of the figures column, above the hero.
export function tryThis(parent, text) {
  const div = document.createElement('div');
  div.className = 'ui-trythis';
  const icon = document.createElement('span');
  icon.className = 'ui-trythis-icon';
  icon.innerHTML =
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
    '<path d="M8 1l1.6 5.4L15 8l-5.4 1.6L8 15 6.4 9.6 1 8l5.4-1.6z" fill="currentColor"/></svg>';
  const p = document.createElement('span');
  p.textContent = text;
  div.append(icon, p);
  parent.insertBefore(div, parent.firstChild);
  return div;
}

// AMENDMENT 5: a one-line narration caption, driven by the model state.
export function narration(parent) {
  const p = document.createElement('p');
  p.className = 'ui-narration';
  parent.appendChild(p);
  return {
    set(text) {
      p.textContent = text;
    },
    get() {
      return p.textContent;
    },
  };
}

const ui = {
  layout, section, slider, number, select, toggle, button,
  plot, raster, heatmap, canvas, hero, narration, tryThis,
  note, equation, code,
  rng, color, onTheme, animate, player, live, ease,
  setTheme, getTheme, keyedParams,
};
export default ui;
