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

  const cellsCtl = ui.number(controls, { key: 'cells_per_column', label: 'cells_per_column', unit: 'cells', value: TemporalMemoryDefaults.cells_per_column, onInput: () => reset() });
  const actThCtl = ui.number(controls, { key: 'activation_threshold', label: 'activation_threshold', unit: 'synapses', value: 3, onInput: () => reset() });
  const learnThCtl = ui.number(controls, { key: 'learning_threshold', label: 'learning_threshold', unit: 'synapses', value: 2, onInput: () => reset() });
  const maxNewCtl = ui.number(controls, { key: 'max_new_synapses', label: 'max_new_synapses', unit: 'synapses', value: 5, onInput: () => reset() });
  const incCtl = ui.slider(controls, { key: 'permanence_increment', label: 'permanence_increment', unit: 'a.u.', min: 0.01, max: 0.3, step: 0.01, value: 0.12, onInput: () => reset() });
  const decCtl = ui.slider(controls, { key: 'permanence_decrement', label: 'permanence_decrement', unit: 'a.u.', min: 0.0, max: 0.2, step: 0.01, value: 0.04, onInput: () => reset() });
  const seqCtl = ui.select(controls, { key: 'sequence', label: 'sequence', options: Object.keys(SEQUENCES).map((k) => ({ value: k, label: k })), value: 'A B C D X B C Y', onInput: () => reset() });
  const learnToggle = ui.toggle(controls, { key: 'learn', label: 'learning on', value: true, onInput: () => {} });

  const stepBtn = ui.button(controls, { label: 'Step', onClick: () => stepOnce() });
  const playBtn = ui.button(controls, { label: 'Play', onClick: () => togglePlay() });
  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => reset() });

  const gridCanvas = ui.canvas(figures, { height: 220 });
  const anomalyPlot = ui.plot(figures, { height: 180, xLabel: 'step', yLabel: 'anomaly score', xUnit: 'steps', yUnit: 'a.u.' });
  // The canvas backing store resets on layout/resize, so register the draw via
  // onResize (called after layout and on every resize) rather than once.
  gridCanvas.onResize(drawGrid);

  let tm = null;
  let seq = [];
  let pos = 0;
  let playing = false;
  let anim = null;
  let anomalyHistory = [];
  let lastSymbol = null;
  let accMs = 0;
  let booting = false;
  const STEP_MS = 500; // ~2 symbols per second while playing

  function reset() {
    stopPlay();
    tm = new TemporalMemory({
      n_columns: 18,
      cells_per_column: cellsCtl.get(),
      activation_threshold: actThCtl.get(),
      learning_threshold: learnThCtl.get(),
      max_new_synapses: maxNewCtl.get(),
      permanence_increment: incCtl.get(),
      permanence_decrement: decCtl.get(),
    });
    seq = SEQUENCES[seqCtl.get()];
    pos = 0;
    anomalyHistory = [];
    lastSymbol = null;
    accMs = 0;
    // Default learning run (8 epochs) so the anomaly curve and grid show real
    // content on load; Step/Play continue from here. booting suppresses the
    // notes until the run finishes so they are not appended every step.
    booting = true;
    for (let e = 0; e < 8; e++) for (let i = 0; i < seq.length; i++) stepOnce();
    booting = false;
    updateNotes();
  }

  function stepOnce() {
    if (!tm) return;
    const sym = seq[pos % seq.length];
    lastSymbol = sym;
    tm.compute(SYMBOLS[sym], learnToggle.get());
    anomalyHistory.push({ step: pos, symbol: sym, anomaly: tm.anomaly_score });
    pos += 1;
    redraw();
  }

  function togglePlay() {
    if (playing) {
      stopPlay();
      playBtn.textContent = 'Play';
    } else {
      playing = true;
      playBtn.textContent = 'Stop';
      accMs = 0;
      anim = ui.animate((dtMs) => {
        accMs += dtMs;
        while (accMs >= STEP_MS) {
          accMs -= STEP_MS;
          stepOnce();
        }
      });
      anim.start();
    }
  }

  function stopPlay() {
    playing = false;
    if (anim) {
      anim.stop();
      anim = null;
    }
  }

  function redraw() {
    if (!tm) return;
    drawGrid();
    drawAnomaly();
    if (!booting) updateNotes();
  }

  function drawGrid() {
    const { ctx, width, height } = gridCanvas;
    ctx.clearRect(0, 0, width, height);
    const nCols = 18;
    const nCells = tm.cells_per_column;
    const pad = 6;
    const cellW = (width - 2 * pad) / nCols;
    const cellH = (height - 2 * pad) / nCells;
    const c1 = ui.color('c1'); // active
    const c3 = ui.color('c3'); // predictive
    const c2 = ui.color('c2'); // bursting column
    const muted = ui.color('muted');
    for (let c = 0; c < nCols; c++) {
      const bursting = tm.bursting_columns[c];
      for (let k = 0; k < nCells; k++) {
        const cell = c * nCells + k;
        const x = pad + c * cellW;
        const y = pad + (nCells - 1 - k) * cellH;
        let fill = muted;
        if (tm.predictive_cells[cell]) fill = c3;
        if (tm.active_cells[cell]) fill = c1;
        ctx.fillStyle = fill;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
        if (bursting) {
          ctx.strokeStyle = c2;
          ctx.strokeRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
        }
      }
    }
  }

  function drawAnomaly() {
    const xs = anomalyHistory.map((h) => h.step);
    const ys = anomalyHistory.map((h) => h.anomaly);
    anomalyPlot.draw([{ x: xs, y: ys, label: 'anomaly', color: 'c2', style: 'line' }], {
      yRange: [0, 1],
    });
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student steps through.
  let dynNotes = null;

  function updateNotes() {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const nBurst = tm.bursting_columns.filter(Boolean).length;
    ui.note(
      dynNotes,
      `<p>Temporal Memory learns sequences of column SDRs. Each column holds <b>${tm.cells_per_column}</b> cells; a cell is ` +
        `<b>predictive</b> when one of its dendritic segments has ≥ <b>activation_threshold = ${actThCtl.get()}</b> connected synapses ` +
        `on currently active cells. On the next step, a column activates only its predicted cells if any are predicted, otherwise it ` +
        `<b>bursts</b> (all cells fire). This step: symbol <b>${lastSymbol ?? '—'}</b>, <b>${nBurst}</b> bursting columns, anomaly ` +
        `<b>${tm.anomaly_score.toFixed(2)}</b> (fraction of active columns that burst).</p>`
    );
    ui.code(dynNotes, () =>
      `from conex import TemporalMemory, TemporalMemoryConfig\n` +
      `cfg = TemporalMemoryConfig(n_columns=18, cells_per_column=${tm.cells_per_column},\n` +
      `    activation_threshold=${actThCtl.get()}, learning_threshold=${learnThCtl.get()},\n` +
      `    max_new_synapses=${maxNewCtl.get()}, permanence_increment=${incCtl.get()},\n` +
      `    permanence_decrement=${decCtl.get()})\n` +
      `tm = TemporalMemory(config=cfg)\n` +
      `for active_columns in sequence:\n` +
      `    tm.compute(neurons, active_columns, learn=${learnToggle.get()})`
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
  reset();
  ui.onTheme(redraw);
  return { destroy() { stopPlay(); } };
}
