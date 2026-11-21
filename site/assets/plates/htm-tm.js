import { TemporalMemory, TemporalMemoryDefaults } from '../js/models/htm.js';
import { ease } from '../js/core.js';

export const meta = {
  id: 'htm-tm',
  title: 'Temporal Memory',
  conex: ['TemporalMemory', 'TemporalMemoryConfig', 'SegmentStorage'],
  source: 'conex/behaviors/neurons/numenta/sequence_memory.py',
};

// Map a set of active column indices to the symbol whose columns they are.
function symbolOfColumns(activeCols) {
  for (const [sym, cols] of Object.entries(SYMBOLS)) {
    if (cols.length === activeCols.length && cols.every((c) => activeCols.includes(c))) return sym;
  }
  return '?';
}

// Pure hero state for the Temporal Memory plate. The prediction shown before
// this step's input is the one computed at the previous step. The eased
// progress f drives the two-phase motion: the predictive cells glow in the
// first half, then the input lights the active cells and bursting columns
// flash in the second half.
export function tmHeroState(run, t) {
  const N = run.steps.length;
  const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
  const f = ease(t - idx);
  const st = run.steps[idx];
  const prev = idx > 0 ? run.steps[idx - 1] : null;
  const prediction = prev ? prev.predictive : new Array(run.nCells).fill(false);
  const nBurst = st.bursting.filter(Boolean).length;
  const nPred = prediction.filter(Boolean).length;
  // Current symbol from the active columns; predicted symbols from the
  // prediction's columns.
  const activeCols = [];
  for (let c = 0; c < run.nCols; c++) if (st.active[c * run.cells]) activeCols.push(c);
  const currentSymbol = symbolOfColumns(activeCols);
  const predCols = new Set();
  for (let i = 0; i < prediction.length; i++) if (prediction[i]) predCols.add(Math.floor(i / run.cells));
  const predSymbols = Object.keys(SYMBOLS).filter((s) => SYMBOLS[s].some((c) => predCols.has(c)));
  const narration =
    `step ${idx + 1}/${N} (${st.symbol}): ${nPred} predicted cells glow, then the input arrives — ` +
    `${nBurst} column${nBurst === 1 ? '' : 's'} burst, anomaly ${st.anomaly.toFixed(2)}`;
  return {
    idx, f, prediction, active: st.active, bursting: st.bursting, anomaly: st.anomaly,
    currentSymbol, predSymbols, narration,
  };
}

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

  // Hero: the columns x cells grid. Predicted cells glow before the input
  // arrives, then the input lights the columns; segment links run from the
  // current winners to the previous winners.
  const hero = ui.hero(figures, { height: 240 });
  const heroCap = { set(text) { hero.narration(text); } };
  const anomalyPlot = ui.plot(figures, { height: 180, xLabel: 'step', yLabel: 'anomaly score', xUnit: 'steps', yUnit: 'a.u.' });
  hero.onResize(() => { if (run) drawGrid(); });

  let run = null; // precomputed replay run (state recorded per step)
  let lastT = 0; // current player time, in steps

  // The player replays the recorded steps one by one. integer:false so
  // onFrame gets the fractional time and the hero eases between steps.
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'steps', playSeconds: 5, loop: true, integer: false, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const N = run.steps.length;
    const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
    drawGrid();
    anomalyPlot.draw([{ x: run.steps.map((_, i) => i), y: run.steps.map((s) => s.anomaly), label: 'anomaly', color: 'c2', style: 'line' }], {
      yRange: [0, 1],
      revealX: idx,
    });
    updateNotes(idx);
  }

  // Hero: the current symbol large on the left, the predicted next symbol(s),
  // and the columns as vertical stacks of small circles. Inactive cells are
  // very faint; active cells are teal; predictive cells get an ochre glowing
  // ring in the first half of the step; bursting columns flash coral in the
  // second half, then settle.
  function drawGrid() {
    const st = tmHeroState(run, lastT);
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const pad = 8;
    const nCols = run.nCols;
    const nCells = run.cells;
    const c1 = ui.color('c1'); // active
    const c3 = ui.color('c3'); // predictive
    const c2 = ui.color('c2'); // bursting
    const muted = ui.color('muted');
    const ink = ui.color('ink');

    // Left panel: the current symbol and the predicted next symbol(s).
    const panelW = Math.min(150, width * 0.22);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('now', pad, pad);
    ctx.fillStyle = ink;
    ctx.font = 'bold 40px ui-serif, Georgia, serif';
    ctx.fillText(st.currentSymbol, pad, pad + 14);
    ctx.fillStyle = muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('predicts', pad, pad + 66);
    ctx.fillStyle = c3;
    ctx.font = 'bold 20px ui-serif, Georgia, serif';
    ctx.fillText(st.predSymbols.length ? st.predSymbols.join(' / ') : 'nothing', pad, pad + 80);

    // Columns: vertical stacks of small circles to the right of the panel.
    const gridX = panelW + pad;
    const gridW = width - gridX - pad;
    const colW = gridW / nCols;
    const cellR = Math.min(colW * 0.42, (height - 2 * pad) / nCells * 0.42);
    // Two-phase motion: predictive glow in the first half, burst flash in the
    // second half, both eased over the step.
    const glow = st.f < 0.5 ? ease(st.f * 2) : 1;
    const flash = st.f < 0.5 ? 0 : ease((st.f - 0.5) * 2);

    for (let c = 0; c < nCols; c++) {
      const bursting = st.bursting[c];
      const cx = gridX + c * colW + colW / 2;
      for (let k = 0; k < nCells; k++) {
        const cell = c * nCells + k;
        const cy = pad + (nCells - 1 - k) * ((height - 2 * pad) / nCells) + (height - 2 * pad) / nCells / 2;
        const active = st.active[cell];
        const predicted = st.prediction[cell];
        // Inactive cells stay quiet (very faint) so the active ones stand out.
        if (!active && !predicted) {
          ctx.globalAlpha = 0.12;
          ctx.fillStyle = muted;
          ctx.beginPath();
          ctx.arc(cx, cy, cellR, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
          continue;
        }
        ctx.fillStyle = active ? c1 : muted;
        ctx.beginPath();
        ctx.arc(cx, cy, cellR, 0, Math.PI * 2);
        ctx.fill();
        // Predictive cells: an ochre glowing ring (brightest in the first half).
        if (predicted) {
          ctx.strokeStyle = c3;
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.4 + 0.6 * glow;
          ctx.beginPath();
          ctx.arc(cx, cy, cellR + 2, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        // Bursting columns: the whole column flashes coral in the second half.
        if (bursting && flash > 0) {
          ctx.strokeStyle = c2;
          ctx.lineWidth = 2;
          ctx.globalAlpha = flash;
          ctx.beginPath();
          ctx.arc(cx, cy, cellR + 3, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }
    heroCap.set(st.narration);
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
          winners: tm.winner_cells.slice(),
          anomaly: tm.anomaly_score,
        });
      }
    }
    run = { cells: cellsCtl.get(), nCells: tm.n_cells, nCols: tm.n_columns, steps };
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
