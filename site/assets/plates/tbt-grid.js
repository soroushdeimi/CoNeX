import { GridCellModule } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-grid',
  title: 'Grid cell path integration',
  conex: ['GridCellModule'],
  source: 'conex/behaviors/neurons/numenta/grid_cells.py',
};

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// Fading trail: alpha 1 at age 0, 0 at age = window, clamped.
export function trailAlpha(age, window) {
  if (window <= 0) return 0;
  return Math.max(0, Math.min(1, 1 - age / window));
}

// Glow radius (px) around the agent, driven by the mean grid activation (0..1 a.u.).
export function glowRadius(meanActivation) {
  return 10 + 30 * Math.max(0, Math.min(1, meanActivation));
}

// One-line narration driven by the model state.
export function gridNarration({ meanAct, speed }) {
  if (speed === 0) return 'agent at rest: the grid cells hold the position code';
  const level = meanAct > 2 / 3 ? 'high' : meanAct > 1 / 3 ? 'medium' : 'low';
  return `agent walks at ${speed.toFixed(2)} a.u./s; grid cells at its position fire at ${level} level (mean ${meanAct.toFixed(2)} a.u.)`;
}

// The model extent shown (a.u.) and the hero rate-map resolution (samples per
// axis). Exported so the tests can check the sampling against the wavelength.
export const SPAN = 4;
export const HERO_N = 80;

// Sample spacing (a.u.) of a rate map: the model extent divided by the number
// of samples per axis. It must be <= lambda/8 (>= 8 samples per spatial
// wavelength) or the hexagonal pattern aliases into a checkerboard.
export function rateMapSpacing(span, n) {
  if (n <= 1) return 0;
  return span / (n - 1);
}

// The 6-fold symmetry residual of a grid cell's rate map: the max, over the
// given radii and angles, of |rate(c + r*u(theta)) - rate(c + r*u(theta+step))|.
// A true hexagonal map is invariant under a 60-degree rotation about a field
// center, so this is ~0 (float noise). A wrong symmetry (e.g. 90 deg) is large.
export function gridSymmetryDiff(rate, center, radii, angles, stepDeg = 60) {
  const step = (stepDeg * Math.PI) / 180;
  let max = 0;
  for (const r of radii) {
    for (const th of angles) {
      const a = rate(center[0] + r * Math.cos(th), center[1] + r * Math.sin(th));
      const b = rate(center[0] + r * Math.cos(th + step), center[1] + r * Math.sin(th + step));
      const d = Math.abs(a - b);
      if (d > max) max = d;
    }
  }
  return max;
}

// Activation of one cell (mirrors GridCellModule.computeActivation for a single cell).
function cellActivation(module, cell, x, y) {
  const px = x - module.gridPhase[cell * 2];
  const py = y - module.gridPhase[cell * 2 + 1];
  let sum = 0;
  for (let i = 0; i < 3; i++) {
    sum += Math.cos(module.waveVectors[(cell * 3 + i) * 2] * px + module.waveVectors[(cell * 3 + i) * 2 + 1] * py);
  }
  return (sum / 3 + 1) / 2;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- model parameters (CoNeX names and defaults) ---
  const nModules = 4;
  const cellsPerModule = 25;

  const sec = ui.section(controls, 'GridCellModule');
  const scaleMin = ui.slider(sec, {
    key: 'scaleMin', label: 'scale_range[0]', unit: 'a.u.',
    min: 0.2, max: 1.0, step: 0.05, value: 0.5,
  });
  const scaleMax = ui.slider(sec, {
    key: 'scaleMax', label: 'scale_range[1]', unit: 'a.u.',
    min: 1.0, max: 4.0, step: 0.05, value: 2.0,
  });
  const orientMax = ui.slider(sec, {
    key: 'orientMax', label: 'orientation_range[1]', unit: 'rad',
    min: 0, max: Math.PI / 2, step: 0.01, value: Math.PI / 3,
  });
  const speed = ui.slider(sec, {
    key: 'speed', label: 'walk speed', unit: 'a.u./s',
    min: 0, max: 2, step: 0.05, value: 0.6,
  });
  const turn = ui.slider(sec, {
    key: 'turnRate', label: 'turn rate', unit: 'rad/s',
    min: 0, max: 2, step: 0.05, value: 0.5,
  });
  const autoWalk = ui.toggle(sec, { key: 'autoWalk', label: 'auto-walk', value: true });
  const heroModule = ui.select(sec, {
    key: 'heroModule', label: 'hero firing map (module)',
    options: [0, 1, 2, 3].map((m) => ({ value: String(m), label: `module ${m}` })),
    value: '0',
  });

  // Fixed phases so the pattern is reproducible (torch.rand in CoNeX).
  const phases = buildPhases(nModules, cellsPerModule, 1234);
  let module = buildModule();
  function buildModule() {
    return new GridCellModule({
      nModules, cellsPerModule,
      scaleRange: [scaleMin.get(), scaleMax.get()],
      orientationRange: [0.0, orientMax.get()],
      phases,
    });
  }

  // --- figures: hero arena on top, one small firing map per module below ---
  ui.note(figures, '<em>Try this: raise <code>turn rate</code> and watch the agent spiral &mdash; the glow brightens where the agent sits on a bright patch of the firing-rate map.</em>');
  const hero = ui.hero(figures, { height: 300 });
  const maps = ui.canvas(figures, { height: 220 });

  const PAD = 20;
  const DT = 0.05; // s of model time per step
  const WINDOW = 5; // s of trail kept on screen

  let heading = 0;
  let t = 0; // model time (s)
  let trail = []; // {x, y, t}
  let meanAct = 0;

  function pos() { return module.gridPosition; }

  function reset() {
    module.gridPosition = [1, 1];
    heading = 0;
    t = 0;
    trail = [];
  }

  // One dt of the model: turn, path-integrate, wrap, record the trail.
  function step() {
    if (!autoWalk.get()) return;
    t += DT;
    heading += turn.get() * DT;
    const v = speed.get();
    module.pathIntegrate([v * Math.cos(heading), v * Math.sin(heading)], DT);
    let [x, y] = module.gridPosition;
    if (x > SPAN / 2) x -= SPAN;
    if (x < -SPAN / 2) x += SPAN;
    if (y > SPAN / 2) y -= SPAN;
    if (y < -SPAN / 2) y += SPAN;
    module.gridPosition = [x, y];
    trail.push({ x, y, t });
    while (trail.length && trail[0].t < t - WINDOW) trail.shift();
  }

  function drawHero() {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    // Equal x/y scale: the model extent (SPAN x SPAN, square) is fitted into the
    // canvas height and centred, so the firing map has square pixels in model
    // units (the real geometry, not a stretched checkerboard).
    const side = height - 2 * PAD;
    const left = (width - side) / 2;
    const top = PAD;
    const sx = (x) => left + ((x + SPAN / 2) / SPAN) * side;
    const sy = (y) => top + ((SPAN / 2 - y) / SPAN) * side;
    // Background: the REAL firing-rate map of ONE cell of the selected module
    // (the same quantity as the small map below it), sampled at >= 8 samples per
    // spatial wavelength so the hexagonal pattern is not aliased into a
    // checkerboard. Drawn at low opacity so the agent passes through firing
    // fields. No square lattice lines.
    const mIdx = Number(heroModule.get());
    const cell = mIdx * cellsPerModule;
    const N = HERO_N;
    const cw = side / N;
    for (let r = 0; r < N; r++) {
      const wy = -SPAN / 2 + (r / (N - 1)) * SPAN;
      for (let c = 0; c < N; c++) {
        const wx = -SPAN / 2 + (c / (N - 1)) * SPAN;
        const a = cellActivation(module, cell, wx, wy);
        if (a > 0.03) {
          ctx.globalAlpha = 0.35 * a;
          ctx.fillStyle = c1;
          ctx.fillRect(left + c * cw, top + r * cw, cw + 0.5, cw + 0.5);
        }
      }
    }
    ctx.globalAlpha = 1;
    // light frame + axis scale in a.u. (unit in an axis title, not a tick)
    ctx.strokeStyle = muted;
    ctx.strokeRect(left, top, side, side);
    ctx.fillStyle = muted;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('-2', left - 2, top + side + 14);
    ctx.fillText('0', left + side / 2 - 3, top + side + 14);
    ctx.fillText('+2', left + side - 12, top + side + 14);
    ctx.fillText('position (a.u.)', left + side / 2 - 34, top + side + 28);
    // caption: name the cell whose rate map is shown
    ctx.fillStyle = ui.color('ink');
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`firing-rate map: module ${mIdx}, cell ${cell}`, left, top - 6);
    // fading trail (skip segments that cross the wrap)
    ctx.strokeStyle = c1;
    ctx.lineWidth = 2;
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      if (Math.abs(a.x - b.x) > SPAN / 2 || Math.abs(a.y - b.y) > SPAN / 2) continue;
      ctx.globalAlpha = 0.5 * trailAlpha(t - b.t, WINDOW);
      ctx.beginPath(); ctx.moveTo(sx(a.x), sy(a.y)); ctx.lineTo(sx(b.x), sy(b.y)); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    // firing-field glow: concentric fills, radius driven by the mean activation
    const [x, y] = pos();
    const r = glowRadius(meanAct);
    ctx.fillStyle = c1;
    for (let i = 4; i >= 1; i--) {
      ctx.globalAlpha = 0.08 * i;
      ctx.beginPath(); ctx.arc(sx(x), sy(y), (r * i) / 4, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // agent
    ctx.fillStyle = c1;
    ctx.beginPath();
    ctx.arc(sx(x), sy(y), 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = c1;
    ctx.beginPath();
    ctx.moveTo(sx(x), sy(y));
    ctx.lineTo(sx(x) + 14 * Math.cos(heading), sy(y) - 14 * Math.sin(heading));
    ctx.stroke();
    hero.narration(gridNarration({ meanAct, speed: speed.get() }));
  }

  // One small firing map per module (cell 0 of each), with the agent's position.
  function drawMaps() {
    const { ctx, width, height } = maps;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const n = nModules;
    const side = Math.min((width - (n + 1) * 8) / n, height - 26);
    const [ax, ay] = pos();
    for (let m = 0; m < n; m++) {
      const x0 = 8 + m * (side + 8);
      const y0 = 8;
      const cell = m * cellsPerModule;
      const N = 48;
      const cw = side / N;
      for (let r = 0; r < N; r++) {
        const wy = -SPAN / 2 + (r / (N - 1)) * SPAN;
        for (let c = 0; c < N; c++) {
          const wx = -SPAN / 2 + (c / (N - 1)) * SPAN;
          const a = cellActivation(module, cell, wx, wy);
          if (a > 0.02) {
            ctx.globalAlpha = a;
            ctx.fillStyle = c1;
            ctx.fillRect(x0 + c * cw, y0 + r * cw, cw + 0.5, cw + 0.5);
          }
        }
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = muted;
      ctx.strokeRect(x0, y0, side, side);
      // agent position on this module's map
      const dx = x0 + ((ax + SPAN / 2) / SPAN) * side;
      const dy = y0 + ((SPAN / 2 - ay) / SPAN) * side;
      ctx.fillStyle = c2;
      ctx.beginPath(); ctx.arc(dx, dy, 3, 0, 2 * Math.PI); ctx.fill();
      // Two lines per caption so it fits the narrow map (a single line overflows
      // into the neighbouring map at 390px).
      ctx.fillStyle = ink;
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`module ${m}`, x0, y0 + side + 12);
      ctx.fillText(`λ = ${module.scales[cell].toFixed(2)} a.u.`, x0, y0 + side + 24);
    }
  }

  function render() {
    const act = module.computeActivation(pos());
    let s = 0;
    for (const v of act) s += v;
    meanAct = s / act.length;
    drawHero();
    drawMaps();
  }

  // Module changes rebuild the firing maps and reset the walk.
  function rebuild() {
    module = buildModule();
    reset();
    render();
  }
  scaleMin.onInput(rebuild);
  scaleMax.onInput(rebuild);
  orientMax.onInput(rebuild);
  // speed / turn rate / auto-walk apply on the next step without a reset
  speed.onInput(render);
  turn.onInput(render);
  autoWalk.onInput(render);
  heroModule.onInput(render);

  // The live runner steps the real model each frame (path integration).
  const live = ui.live(figures, {
    dt: DT, unit: 's', modelPerSecond: 1, windowSize: WINDOW,
    step, reset,
    onFrame: render,
  });

  // drag to move the agent
  const canvas = hero.canvas;
  let dragging = false;
  function toWorld(ev) {
    const rect = canvas.getBoundingClientRect();
    // match the hero's equal-scale, centred mapping (side = height - 2*PAD)
    const side = rect.height - 2 * PAD;
    const left = (rect.width - side) / 2;
    const x = ((ev.clientX - rect.left - left) / side) * SPAN - SPAN / 2;
    const y = SPAN / 2 - ((ev.clientY - rect.top - PAD) / side) * SPAN;
    return [x, y];
  }
  const onDown = (ev) => { dragging = true; autoWalk.set(false); };
  const onMove = (ev) => {
    if (!dragging) return;
    const [x, y] = toWorld(ev);
    module.gridPosition = [x, y];
    trail.push({ x, y, t });
    render();
  };
  const onUp = () => { dragging = false; };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);

  // Draw inside onResize (called after layout and on every resize); drawing
  // once at mount with a 0 width would leave the canvases empty.
  hero.onResize(render);
  maps.onResize(render);
  ui.onTheme(render);
  render();

  // --- notes ---
  ui.note(notes, `
    <p>Grid cells fire when an agent sits on the vertices of a hexagonal lattice.
    Each <em>module</em> has a characteristic scale (spacing) and orientation;
    CoNeX assigns the modules logarithmically spaced scales and evenly spaced
    orientations. The firing rate is the sum of three cosines whose wave vectors
    sit 60&deg; apart, which is what produces the hexagon.</p>
    <p>Position is maintained by <strong>path integration</strong>: the module
    integrates velocity over time (dead reckoning). This is the allocentric
    reference frame the Thousand Brains Theory uses to let each column track
    where an object is.</p>
    <p>The hero shows the agent walking with a fading trail (the last 5 s of
    model time) over the <em>real firing-rate map of the selected module</em>
    (the same data as the small maps below, at low opacity). The glow around the
    agent is the <em>mean activation of all grid cells at its position</em>
    &mdash; it brightens where the agent sits on a bright patch of the map. The
    four small maps are the firing fields of one cell per module; the dot is the
    agent's position on each map.</p>
    <p><strong>Important:</strong> the hexagonal periodicity is <em>built into the
    model</em>. The wave vectors are fixed by the scale and orientation, so the
    lattice is imposed by construction &mdash; it is not learned from experience.
    Only the per-cell phase offsets are random.</p>
  `);
  ui.equation(notes, `
    <p><var>r</var>(<var>x</var>) = (1/3) &sum;<sub>i=1..3</sub> cos(<var>k</var><sub>i</sub> &middot; (<var>x</var> &minus; <var>phase</var>))</p>
    <p><var>k</var><sub>i</sub> = (2&pi;/&lambda;) [cos(&theta; + i&middot;60&deg;), sin(&theta; + i&middot;60&deg;)], &nbsp; activation = (<var>r</var> + 1)/2</p>
    <p>path integration: <var>x</var> &larr; <var>x</var> + <var>v</var> &middot; <var>dt</var></p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> a grid module is like a bank of sinusoidal
    feature detectors (a Fourier basis) at different frequencies. The analogy
    breaks because the frequencies and angles are fixed by the module's scale and
    orientation rather than learned, and the code is a continuous position, not a
    discrete class.</p>
  `);

  // --- CoNeX code box ---
  ui.code(notes, () => `
import torch
from pymonntorch import Network, NeuronGroup
from conex import GridCellModule, TimeResolution

module = GridCellModule(
    n_modules=${nModules},
    cells_per_module=${cellsPerModule},
    scale_range=(${fmt(scaleMin.get())}, ${fmt(scaleMax.get())}),
    orientation_range=(0.0, ${fmt(orientMax.get())}),
)
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(
    net=net, size=${nModules} * ${cellsPerModule}, behavior={100: module}, tag="grid",
)
net.initialize()

# path integration: position += velocity * dt
module.path_integrate(neurons, torch.tensor([1.0, 0.0]), dt=0.05)
activation = module.encode_position(neurons, torch.tensor([1.0, 1.0]))
`);

  return {
    destroy() {
      live.pause();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    },
  };
}

// At most 4 significant digits for numbers shown in text and code snippets.
function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  return Number(v.toPrecision(4)).toString();
}

// Deterministic 2D phases in [0, 2*pi) so the pattern is reproducible.
function buildPhases(nModules, cellsPerModule, seed) {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float64Array(nModules * cellsPerModule * 2);
  for (let i = 0; i < out.length; i++) out[i] = rand() * 2 * Math.PI;
  return out;
}
