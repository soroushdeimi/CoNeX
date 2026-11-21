import { GridCellModule } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-grid',
  title: 'Grid cell path integration',
  conex: ['GridCellModule'],
  source: 'conex/behaviors/neurons/numenta/grid_cells.py',
};

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
  const autoWalk = ui.toggle(sec, { key: 'autoWalk', label: 'auto-walk', value: true });

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

  // --- figures ---
  const arena = ui.canvas(figures, { height: 320 });
  const firing = ui.heatmap(figures, { rows: 40, cols: 40, label: 'firing map, module 0 cell 0' });

  const pos = [1.0, 1.0];
  let heading = 0.0;

  const SPAN = 6;
  const PAD = 20;

  function drawFiring() {
    const values = [];
    const N = 40;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = -SPAN / 2 + (c / (N - 1)) * SPAN;
        const y = -SPAN / 2 + (r / (N - 1)) * SPAN;
        values.push(module.computeActivation([x, y])[0]);
      }
    }
    firing.draw(values, { min: 0, max: 1 });
  }

  function rebuild() {
    module = buildModule();
    module.gridPosition = [pos[0], pos[1]];
    module.forward();
    drawFiring(); // firing map depends only on the module, not the position
  }
  scaleMin.onInput(rebuild);
  scaleMax.onInput(rebuild);
  orientMax.onInput(rebuild);
  drawFiring(); // initial firing map

  function drawArena() {
    const { ctx, width, height } = arena;
    ctx.clearRect(0, 0, width, height);
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const sx = (x) => PAD + ((x + SPAN / 2) / SPAN) * (width - 2 * PAD);
    const sy = (y) => height - PAD - ((y + SPAN / 2) / SPAN) * (height - 2 * PAD);
    ctx.strokeStyle = muted;
    ctx.globalAlpha = 0.25;
    for (let g = -3; g <= 3; g++) {
      ctx.beginPath(); ctx.moveTo(sx(g), sy(-3)); ctx.lineTo(sx(g), sy(3)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(-3), sy(g)); ctx.lineTo(sx(3), sy(g)); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = c1;
    ctx.beginPath();
    ctx.arc(sx(pos[0]), sy(pos[1]), 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = c1;
    ctx.beginPath();
    ctx.moveTo(sx(pos[0]), sy(pos[1]));
    ctx.lineTo(sx(pos[0]) + 14 * Math.cos(heading), sy(pos[1]) - 14 * Math.sin(heading));
    ctx.stroke();
  }

  function render() {
    drawArena();
  }

  const anim = ui.animate((dtMs) => {
    if (autoWalk.get()) {
      heading += 0.15 * (dtMs / 16.7);
      const v = speed.get();
      module.pathIntegrate([v * Math.cos(heading), v * Math.sin(heading)], dtMs / 1000);
      pos[0] = module.gridPosition[0];
      pos[1] = module.gridPosition[1];
      if (pos[0] > 3) pos[0] = -3;
      if (pos[0] < -3) pos[0] = 3;
      if (pos[1] > 3) pos[1] = -3;
      if (pos[1] < -3) pos[1] = 3;
    }
    render();
  });
  anim.start();

  // drag to move the agent
  const canvas = arena.canvas;
  let dragging = false;
  function toWorld(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left - PAD) / (rect.width - 2 * PAD)) * SPAN - SPAN / 2;
    const y = SPAN / 2 - ((ev.clientY - rect.top - PAD) / (rect.height - 2 * PAD)) * SPAN;
    return [x, y];
  }
  const onDown = (ev) => { dragging = true; autoWalk.set(false); };
  const onMove = (ev) => {
    if (!dragging) return;
    const [x, y] = toWorld(ev);
    pos[0] = x; pos[1] = y;
    module.gridPosition = [x, y];
    render();
  };
  const onUp = () => { dragging = false; };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);

  ui.onTheme(render);

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
    <p><strong>Important:</strong> the hexagonal periodicity is <em>built into the
    model</em>. The wave vectors are fixed by the scale and orientation, so the
    lattice is imposed by construction &mdash; it is not learned from experience.
    Only the per-cell phase offsets are random.</p>
  `);
  ui.equation(notes, `
    <p><var>r</var>(<var>x</var>) = (1/3) &sum;<sub>i=1..3</sub> cos(<var>k</var><sub>i</sub> &middot; (<var>x</var> &minus; <var>phase</var>))</p>
    <p><var>k</var><sub>i</sub> = (2&pi;/&lambda;) [cos(&theta; + i&middot;60&deg;), sin(&theta; + i&middot;60&deg;)], &nbsp; activation = (<var>r</var> + 1)/2</p>
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
from conex import GridCellModule

module = GridCellModule(
    n_modules=${nModules},
    cells_per_module=${cellsPerModule},
    scale_range=(${scaleMin.get()}, ${scaleMax.get()}),
    orientation_range=(0.0, ${orientMax.get()}),
)
# path integration: position += velocity * dt
module.path_integrate(neurons, velocity, dt=0.05)
activation = module.encode_position(neurons, position)
`);

  return {
    destroy() {
      anim.stop();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    },
  };
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
