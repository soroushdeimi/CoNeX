import { GridCellModule, DisplacementCellModule } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-displacement',
  title: 'Displacement cells',
  conex: ['DisplacementCellModule', 'LocationEncoder'],
  source: 'conex/behaviors/neurons/numenta/grid_cells.py',
};

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// Arrow length (px) for the A -> agent vector, from the displacement magnitude.
export function arrowLength(magnitude, maxDisplacement) {
  if (maxDisplacement <= 0) return 0;
  return Math.max(0, Math.min(1, magnitude / maxDisplacement)) * 120;
}

// A displacement cell's glow (0..1) is its Gaussian activation.
export function cellGlow(activation) {
  return Math.max(0, Math.min(1, activation));
}

// One-line narration driven by the model state.
export function displacementNarration({ magnitude, decoded, maxDisplacement }) {
  return `agent is ${magnitude.toFixed(2)} a.u. from A; the population decodes the displacement as (${decoded[0].toFixed(2)}, ${decoded[1].toFixed(2)}) a.u. (max ${maxDisplacement.toFixed(1)} a.u.)`;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- displacement module parameters (CoNeX names and defaults) ---
  const sec = ui.section(controls, 'DisplacementCellModule');
  const nCells = ui.number(sec, { key: 'nCells', label: 'n_cells', unit: 'cells', value: 100 });
  const maxDisp = ui.slider(sec, {
    key: 'maxDisplacement', label: 'max_displacement', unit: 'a.u.',
    min: 2, max: 20, step: 0.5, value: 10.0,
  });
  const sigma = ui.slider(sec, {
    key: 'sigma', label: 'sigma', unit: 'a.u.',
    min: 0.1, max: 2, step: 0.05, value: 0.5,
  });

  // --- two locations ---
  const loc = ui.section(controls, 'Locations');
  const ax = ui.slider(loc, { key: 'ax', label: 'A x', unit: 'a.u.', min: -5, max: 5, step: 0.1, value: 1.0 });
  const ay = ui.slider(loc, { key: 'ay', label: 'A y', unit: 'a.u.', min: -5, max: 5, step: 0.1, value: 2.0 });
  const bx = ui.slider(loc, { key: 'bx', label: 'B x', unit: 'a.u.', min: -5, max: 5, step: 0.1, value: 4.0 });
  const by = ui.slider(loc, { key: 'by', label: 'B y', unit: 'a.u.', min: -5, max: 5, step: 0.1, value: 5.0 });

  // Fixed grid phases so the location codes are reproducible.
  const nModules = 4;
  const cellsPerModule = 25;
  const phases = buildPhases(nModules, cellsPerModule, 1234);
  const grid = new GridCellModule({
    nModules, cellsPerModule,
    scaleRange: [0.5, 2.0],
    orientationRange: [0.0, Math.PI / 3],
    phases,
  });

  let disp = buildDisp();
  function buildDisp() {
    return new DisplacementCellModule({
      nCells: nCells.get(),
      maxDisplacement: maxDisp.get(),
      sigma: sigma.get(),
    });
  }

  // Precompute the walk A -> B: at each of N+1 points the agent's position,
  // the real displacement-cell activation for the vector A -> position, and the
  // decoded displacement. The player only indexes into this, so no per-frame
  // recomputation.
  const N = 60;
  let frames = [];
  function precompute() {
    const A = [ax.get(), ay.get()];
    const B = [bx.get(), by.get()];
    frames = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const pos = [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t];
      const act = disp.computeDisplacement(A, pos);
      const decoded = disp.decodeDisplacement(act);
      const magnitude = Math.hypot(pos[0] - A[0], pos[1] - A[1]);
      frames.push({ pos, act, decoded, magnitude });
    }
  }
  const rebuild = () => { disp = buildDisp(); precompute(); render(); };
  nCells.onInput(rebuild);
  maxDisp.onInput(rebuild);
  sigma.onInput(rebuild);
  [ax, ay, bx, by].forEach((s) => s.onInput(rebuild));

  // --- figures: hero arena on top, plots below ---
  const hero = ui.hero(figures, { height: 300 });
  const dispPlot = ui.plot(figures, {
    height: 220, xLabel: 'displacement cell', xUnit: 'index',
    yLabel: 'activation', yUnit: 'a.u.',
  });
  const codePlot = ui.plot(figures, {
    height: 220, xLabel: 'grid cell', xUnit: 'index',
    yLabel: 'activation', yUnit: 'a.u.',
  });

  const SPAN = 12;
  const PAD = 24;
  let currentFrame = null;

  function drawHero() {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const sx = (x) => PAD + ((x + SPAN / 2) / SPAN) * (width - 2 * PAD);
    const sy = (y) => height - PAD - ((y + SPAN / 2) / SPAN) * (height - 2 * PAD);
    const A = [ax.get(), ay.get()];
    const B = [bx.get(), by.get()];
    // faint full A -> B reference arrow
    ctx.strokeStyle = c3;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(sx(A[0]), sy(A[1]));
    ctx.lineTo(sx(B[0]), sy(B[1]));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    // points
    ctx.fillStyle = c1;
    ctx.beginPath(); ctx.arc(sx(A[0]), sy(A[1]), 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = c2;
    ctx.beginPath(); ctx.arc(sx(B[0]), sy(B[1]), 6, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = ink;
    ctx.font = '13px ui-monospace, monospace';
    ctx.fillText('A', sx(A[0]) + 8, sy(A[1]) - 8);
    ctx.fillText('B', sx(B[0]) + 8, sy(B[1]) - 8);

    if (currentFrame) {
      const p = currentFrame.pos;
      const mag = currentFrame.magnitude;
      const ang = Math.atan2(sy(p[1]) - sy(A[1]), sx(p[0]) - sx(A[0]));
      // animated vector arrow A -> agent, length from the real magnitude
      const len = arrowLength(mag, maxDisp.get());
      const tipX = sx(A[0]) + Math.cos(ang) * len;
      const tipY = sy(A[1]) - Math.sin(ang) * len;
      ctx.strokeStyle = c3;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(A[0]), sy(A[1]));
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      ctx.fillStyle = c3;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - 9 * Math.cos(ang - 0.4), tipY + 9 * Math.sin(ang - 0.4));
      ctx.lineTo(tipX - 9 * Math.cos(ang + 0.4), tipY + 9 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.lineWidth = 1;
      // the agent at its current position
      ctx.fillStyle = c1;
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 5, 0, 2 * Math.PI); ctx.fill();

      // polar ring of displacement cells: each cell's glow is its activation
      const cx = width - 90;
      const cy = height / 2;
      const R = 62;
      const n = disp.preferredDx.length;
      for (let c = 0; c < n; c++) {
        const a = currentFrame.act[c];
        const glow = cellGlow(a);
        if (glow < 0.02) continue;
        const th = Math.atan2(disp.preferredDy[c], disp.preferredDx[c]);
        const rr = (Math.hypot(disp.preferredDx[c], disp.preferredDy[c]) / maxDisp.get()) * R;
        const px = cx + Math.cos(th) * rr;
        const py = cy - Math.sin(th) * rr;
        ctx.globalAlpha = glow;
        ctx.fillStyle = c3;
        ctx.beginPath(); ctx.arc(px, py, 2.5 + 3 * glow, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = ink;
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('displacement cells', cx - 44, cy + R + 16);

      hero.narration(displacementNarration({
        magnitude: mag, decoded: currentFrame.decoded, maxDisplacement: maxDisp.get(),
      }));
    }
  }

  function render() {
    drawHero();
    const A = [ax.get(), ay.get()];
    const B = [bx.get(), by.get()];
    const act = currentFrame ? currentFrame.act : disp.computeDisplacement(A, B);
    dispPlot.draw(
      [{ x: Array.from({ length: act.length }, (_, i) => i), y: Array.from(act), label: 'activation', color: 'c3' }],
      { xRange: [0, act.length - 1], yRange: [0, 1] },
    );
    const codeA = grid.computeActivation(A);
    const codeB = grid.computeActivation(B);
    codePlot.draw(
      [
        { x: Array.from({ length: codeA.length }, (_, i) => i), y: Array.from(codeA), label: 'code at A', color: 'c1' },
        { x: Array.from({ length: codeB.length }, (_, i) => i), y: Array.from(codeB), label: 'code at B', color: 'c2' },
      ],
      { xRange: [0, codeA.length - 1], yRange: [0, 1] },
    );
  }

  // The player walks the agent from A to B over 100 a.u. of model time.
  function onFrame(t) {
    const i = Math.max(0, Math.min(N, Math.round((t / 100) * N)));
    currentFrame = frames[i];
    render();
  }

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(render);
  precompute();
  currentFrame = frames[N]; // start at B so the initial draw is the full A->B code
  render();
  ui.onTheme(render);
  ui.player(figures, {
    duration: 100,
    unit: 'a.u.',
    playSeconds: 6,
    loop: true,
    onFrame,
  });

  // --- notes ---
  ui.note(notes, `
    <p>Displacement cells encode the <em>movement between two locations</em>, not
    the locations themselves. Each cell has a preferred displacement vector
    (a magnitude and a direction), and its activation is a Gaussian of the
    distance between the actual displacement and its preferred one. This is how a
    column links a feature seen at one place to the same feature seen at another
    &mdash; the core of learning object structure through movement.</p>
    <p>The hero walks the agent from A to B. The ochre arrow is the <em>real</em>
    displacement vector from A to the agent, its length scaled by the magnitude;
    the polar ring on the right is the displacement-cell population, each cell
    glowing in proportion to its activation for that vector. As the agent moves,
    the arrow grows and the lit cells sweep along the path, settling on the full
    A&rarr;B code at the end.</p>
    <p>The two location codes (bottom plot) are the grid cell activations at A and
    B. The displacement code (middle plot) is the population pattern for the
    vector from A to the agent; a population-vector decode recovers the arrow you
    see in the arena.</p>
  `);
  ui.equation(notes, `
    <p>activation<sub>c</sub> = exp(&minus;|<var>d</var><sub>c</sub> &minus; <var>d</var>|&sup2; / (2&sigma;&sup2;))</p>
    <p>decode: <var>d</var> = &sum;<sub>c</sub> w<sub>c</sub> <var>d</var><sub>c</sub>, &nbsp; w<sub>c</sub> = activation<sub>c</sub> / &sum; activation</p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> a displacement code is like a relative
    position embedding &mdash; it encodes the difference between two tokens rather
    than their absolute positions. The analogy breaks because the preferred
    displacements form a fixed polar grid (magnitudes &times; directions) chosen
    by the module, not a learned embedding space.</p>
  `);

  ui.code(notes, () => `
import torch
from pymonntorch import Network, NeuronGroup
from conex import DisplacementCellModule, GridCellModule, TimeResolution

disp = DisplacementCellModule(
    n_cells=${nCells.get()},
    max_displacement=${fmt(maxDisp.get())},
    sigma=${fmt(sigma.get())},
)
grid = GridCellModule(n_modules=${nModules}, cells_per_module=${cellsPerModule})

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
grid_neurons = NeuronGroup(
    net=net, size=${nModules} * ${cellsPerModule}, behavior={100: grid}, tag="grid",
)
disp_neurons = NeuronGroup(
    net=net, size=${nCells.get()}, behavior={100: disp}, tag="disp",
)
net.initialize()

code_a = grid.encode_position(grid_neurons, torch.tensor([${fmt(ax.get())}, ${fmt(ay.get())}]))
code_b = grid.encode_position(grid_neurons, torch.tensor([${fmt(bx.get())}, ${fmt(by.get())}]))
displacement_code = disp.compute_displacement(
    disp_neurons,
    torch.tensor([${fmt(ax.get())}, ${fmt(ay.get())}]),
    torch.tensor([${fmt(bx.get())}, ${fmt(by.get())}]),
)
decoded = disp.decode_displacement(disp_neurons, displacement_code)
`);

  return { destroy() {} };
}

// At most 4 significant digits for numbers shown in text and code snippets.
function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  return Number(v.toPrecision(4)).toString();
}

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
