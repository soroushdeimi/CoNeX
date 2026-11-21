import { GridCellModule, DisplacementCellModule } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-displacement',
  title: 'Displacement cells',
  conex: ['DisplacementCellModule', 'LocationEncoder'],
  source: 'conex/behaviors/neurons/numenta/grid_cells.py',
};

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
  const rebuild = () => { disp = buildDisp(); render(); };
  nCells.onInput(rebuild);
  maxDisp.onInput(rebuild);
  sigma.onInput(rebuild);
  [ax, ay, bx, by].forEach((s) => s.onInput(render));

  // --- figures ---
  const arena = ui.canvas(figures, { height: 300 });
  const dispPlot = ui.plot(figures, {
    height: 220, xLabel: 'displacement cell index', xUnit: '',
    yLabel: 'activation', yUnit: 'a.u.',
  });
  const codePlot = ui.plot(figures, {
    height: 220, xLabel: 'grid cell index', xUnit: '',
    yLabel: 'activation', yUnit: 'a.u.',
  });

  const SPAN = 12;
  const PAD = 24;

  function drawArena() {
    const { ctx, width, height } = arena;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const sx = (x) => PAD + ((x + SPAN / 2) / SPAN) * (width - 2 * PAD);
    const sy = (y) => height - PAD - ((y + SPAN / 2) / SPAN) * (height - 2 * PAD);
    const A = [ax.get(), ay.get()];
    const B = [bx.get(), by.get()];
    // displacement arrow A -> B
    ctx.strokeStyle = c3;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(A[0]), sy(A[1]));
    ctx.lineTo(sx(B[0]), sy(B[1]));
    ctx.stroke();
    // arrowhead
    const ang = Math.atan2(sy(B[1]) - sy(A[1]), sx(B[0]) - sx(A[0]));
    ctx.beginPath();
    ctx.moveTo(sx(B[0]), sy(B[1]));
    ctx.lineTo(sx(B[0]) - 10 * Math.cos(ang - 0.4), sy(B[1]) - 10 * Math.sin(ang - 0.4));
    ctx.moveTo(sx(B[0]), sy(B[1]));
    ctx.lineTo(sx(B[0]) - 10 * Math.cos(ang + 0.4), sy(B[1]) - 10 * Math.sin(ang + 0.4));
    ctx.stroke();
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
  }

  function render() {
    drawArena();
    const A = [ax.get(), ay.get()];
    const B = [bx.get(), by.get()];
    const act = disp.computeDisplacement(A, B);
    const decoded = disp.decodeDisplacement(act);
    dispPlot.draw(
      [{ x: Array.from({ length: act.length }, (_, i) => i), y: Array.from(act), label: 'activation', color: 'c3' }],
      { xRange: [0, act.length - 1] },
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
    void decoded;
  }
  render();
  ui.onTheme(render);

  // --- notes ---
  ui.note(notes, `
    <p>Displacement cells encode the <em>movement between two locations</em>, not
    the locations themselves. Each cell has a preferred displacement vector
    (a magnitude and a direction), and its activation is a Gaussian of the
    distance between the actual displacement and its preferred one. This is how a
    column links a feature seen at one place to the same feature seen at another
    &mdash; the core of learning object structure through movement.</p>
    <p>The two location codes (top plot) are the grid cell activations at A and
    B. The displacement code (middle plot) is the population pattern for the
    vector from A to B; a population-vector decode recovers the arrow you see in
    the arena.</p>
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
    max_displacement=${maxDisp.get()},
    sigma=${sigma.get()},
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

code_a = grid.encode_position(grid_neurons, torch.tensor([${ax.get()}, ${ay.get()}]))
code_b = grid.encode_position(grid_neurons, torch.tensor([${bx.get()}, ${by.get()}]))
displacement_code = disp.compute_displacement(
    disp_neurons,
    torch.tensor([${ax.get()}, ${ay.get()}]),
    torch.tensor([${bx.get()}, ${by.get()}]),
)
decoded = disp.decode_displacement(disp_neurons, displacement_code)
`);

  return { destroy() {} };
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
