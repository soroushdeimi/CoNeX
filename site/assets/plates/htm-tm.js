import { TemporalMemory, TemporalMemoryDefaults } from '../js/models/htm.js';

export const meta = {
  id: 'htm-tm',
  title: 'Temporal Memory',
  conex: ['TemporalMemory', 'TemporalMemoryConfig', 'SegmentStorage'],
  source: 'conex/behaviors/neurons/numenta/sequence_memory.py',
};

// Six 3-column symbols over 18 columns, matching the golden scenario.
const SYMBOLS = {
  A: [0, 1, 2],
  X: [3, 4, 5],
  B: [6, 7, 8],
  C: [9, 10, 11],
  D: [12, 13, 14],
  Y: [15, 16, 17],
};
const SEQUENCES = {
  'A B C D': ['A', 'B', 'C', 'D'],
  'X B C Y': ['X', 'B', 'C', 'Y'],
  'A B C D X B C Y': ['A', 'B', 'C', 'D', 'X', 'B', 'C', 'Y'],
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const cellsCtl = ui.number(controls, { key: 'cells_per_column', label: 'cells_per_column', unit: 'cells', value: TemporalMemoryDefaults.cells_per_column, onInput: () => recompute() });
  const actThCtl = ui.number(controls, { key: 'activation_threshold', label: 'activation_threshold', unit: 'synapses', value: 3, onInput: () => recompute() });
  const learnThCtl = ui.number(controls, { key: 'learning_threshold', label: 'learning_threshold', unit: 'synapses', value: 2, onInput: () => recompute() });
  const maxNewCtl = ui.number(controls, { key: 'max_new_synapses', label: 'max_new_synapses', unit: 'synapses', value: 5, onInput: () => recompute() });
  const incCtl = ui.slider(controls, { key: 'permanence_increment', label: 'permanence_increment', unit: 'a.u.', min: 0.01, max: 0.3, step: 0.01, value: 0.12, onInput: () => recompute() });
  const decCtl = ui.slider(controls, { key: 'permanence_decrement', label: 'permanence_decrement', unit: 'a.u.', min: 0.0, max: 0.2, step: 0.01, value: 0.04, onInput: () => recompute() });
  const seqCtl = ui.select(controls, { key: 'sequence', label: 'sequence', options: Object.keys(SEQUENCES).map((k) => ({ value: k, label: k })), value: 'A B C D X B C Y', onInput: () => recompute() });
  const learnToggle = ui.toggle(controls, { key: 'learn', label: 'learning on', value: true, onInput: () => {} });

  const stepBtn = ui.button(controls, { label: 'Step', onClick: () => { if (player) player.seek(Math.min(run ? run.steps.length : 0, Math.floor(lastT) + 1)); } });
  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => recompute() });

  const gridCanvas = ui.canvas(figures, { height: 220 });
  const anomalyPlot = ui.plot(figures, { height: 180, xLabel: 'step', yLabel: 'anomaly score', xUnit: 'steps', yUnit: 'a.u.' });
  // Register the redraw on resize when the ui supports it (the real ui.js does);
  // the direct drawGrid() call in frame() covers the no-onResize case.
  if (typeof gridCanvas.onResize === 'function') gridCanvas.onResize(() => { if (run) drawGrid(Math.max(0, Math.min(run.steps.length - 1, Math.floor(lastT)))); });

  let run = null; // precomputed replay run (state recorded per step)
  let lastT = 0; // current player time, in steps

  // The player replays the recorded steps one by one.
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'steps', playSeconds: 5, loop: true, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const N = run.steps.length;
    const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
    drawGrid(idx);
    anomalyPlot.draw([{ x: run.steps.map((_, i) => i), y: run.steps.map((s) => s.anomaly), label: 'anomaly', color: 'c2', style: 'line' }], {
      yRange: [0, 1],
      revealX: idx,
    });
    updateNotes(idx);
  }

  function drawGrid(idx) {
    const st = run.steps[idx];
    const { ctx, width, height } = gridCanvas;
    ctx.clearRect(0, 0, width, height);
    const nCols = 18;
    const nCells = run.cells;
    const pad = 6;
    const cellW = (width - 2 * pad) / nCols;
    const cellH = (height - 2 * pad) / nCells;
    const c1 = ui.color('c1'); // active
    const c3 = ui.color('c3'); // predictive
    const c2 = ui.color('c2'); // bursting column
    const muted = ui.color('muted');
    for (let c = 0; c < nCols; c++) {
      const bursting = st.bursting[c];
      for (let k = 0; k < nCells; k++) {
        const cell = c * nCells + k;
        const x = pad + c * cellW;
        const y = pad + (nCells - 1 - k) * cellH;
        let fill = muted;
        if (st.predictive[cell]) fill = c3;
        if (st.active[cell]) fill = c1;
        ctx.fillStyle = fill;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
        if (bursting) {
          ctx.strokeStyle = c2;
          ctx.strokeRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
        }
      }
    }
  }

  function recompute() {
    const tm = new TemporalMemory({
      n_columns: 18,
      cells_per_column: cellsCtl.get(),
      activation_threshold: actThCtl.get(),
      learning_threshold: learnThCtl.get(),
      max_new_synapses: maxNewCtl.get(),
      permanence_increment: incCtl.get(),
      permanence_decrement: decCtl.get(),
    });
    const seq = SEQUENCES[seqCtl.get()];
    // Boot learning (8 epochs) so the replay starts from a trained memory.
    for (let e = 0; e < 8; e++) for (const sym of seq) tm.compute(SYMBOLS[sym], true);
    // Record 2 epochs of the sequence; the player only indexes into this.
    const steps = [];
    for (let e = 0; e < 2; e++) {
      for (const sym of seq) {
        tm.compute(SYMBOLS[sym], learnToggle.get());
        steps.push({
          symbol: sym,
          active: tm.active_cells.slice(),
          predictive: tm.predictive_cells.slice(),
          bursting: tm.bursting_columns.slice(),
          anomaly: tm.anomaly_score,
        });
      }
    }
    run = { cells: cellsCtl.get(), steps };
    if (player) {
      player.setDuration(steps.length);
      // Keep the current position (clamped) so a parameter change does not restart the replay.
      player.seek(Math.min(lastT, steps.length));
    } else {
      frame(steps.length); // no player available: show the last step
    }
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student steps through.
  let dynNotes = null;

  function updateNotes(idx) {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const st = run.steps[idx];
    const nBurst = st.bursting.filter(Boolean).length;
    ui.note(
      dynNotes,
      `<p>Temporal Memory learns sequences of column SDRs. Each column holds <b>${run.cells}</b> cells; a cell is ` +
        `<b>predictive</b> (ochre) when one of its dendritic segments has ≥ <b>activation_threshold = ${actThCtl.get()}</b> connected synapses ` +
        `on currently active cells. On the next step, a column activates only its predicted cells if any are predicted, otherwise it ` +
        `<b>bursts</b> (all cells fire, coral outline). Step <b>${idx + 1}/${run.steps.length}</b>: symbol <b>${st.symbol}</b>, ` +
        `<b>${nBurst}</b> bursting columns, anomaly <b>${st.anomaly.toFixed(2)}</b> (fraction of active columns that burst).</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from pymonntorch import Network, NeuronGroup\n` +
      `from conex import TemporalMemory, TemporalMemoryConfig, TimeResolution\n` +
      `cfg = TemporalMemoryConfig(n_columns=18, cells_per_column=${run.cells},\n` +
      `    activation_threshold=${actThCtl.get()}, learning_threshold=${learnThCtl.get()},\n` +
      `    max_new_synapses=${maxNewCtl.get()}, permanence_increment=${incCtl.get()},\n` +
      `    permanence_decrement=${decCtl.get()})\n` +
      `net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")\n` +
      `neurons = NeuronGroup(net=net, size=18 * ${run.cells},\n` +
      `    behavior={100: TemporalMemory(config=cfg)}, tag="tm")\n` +
      `net.initialize()\n` +
      `tm = neurons.temporal_memory\n` +
      `def cols(*idx):\n` +
      `    c = torch.zeros(18, dtype=torch.bool)\n` +
      `    c[list(idx)] = True\n` +
      `    return c\n` +
      `sequence = [cols(0, 1, 2), cols(6, 7, 8), cols(9, 10, 11), cols(12, 13, 14)]\n` +
      `for active_columns in sequence:\n` +
      `    tm.compute(neurons, active_columns, learn=${learnToggle.get() ? 'True' : 'False'})`
    );
  }

  function staticNotes() {
    ui.note(
      notes,
      `<p><b>The context story.</b> Learn <b>A-B-C-D</b> and <b>X-B-C-Y</b>. The symbol <b>C</b> is shared, so after A-B-C the memory ` +
        `predicts <b>D</b>, and after X-B-C it predicts <b>Y</b> — the same C leads to different predictions depending on what came ` +
        `before. Cells in a column represent the same input in different contexts, which is how one column can mean "D follows" or ` +
        `"Y follows". Permanences and thresholds are dimensionless (a.u.).</p>`
    );
    ui.equation(
      notes,
      `<div>anomaly = <var>n</var><sub>bursting</sub> / <var>n</var><sub>active columns</sub>, &nbsp; ` +
        `segment active ⇔ <var>Σ</var>[perm ≥ conn]·[target active] ≥ activation_threshold</div>`
    );
    ui.note(
      notes,
      `<p><b>Background.</b> Temporal Memory is the sequence-learning core of HTM: it predicts the next input and flags surprises as ` +
        `bursts. <b>ML analogy:</b> a recurrent network / finite-state predictor. The analogy breaks because learning is local, ` +
        `Hebbian, and context is stored in which cell (not which column) wins — there is no global loss.</p>`
    );
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}
