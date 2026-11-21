import { SpatialPooler, SpatialPoolerDefaults } from '../js/models/htm.js';
import { ease } from '../js/core.js';

export const meta = {
  id: 'htm-sp',
  title: 'Spatial Pooler',
  conex: ['SpatialPooler', 'SpatialPoolerConfig'],
  source: 'conex/behaviors/neurons/numenta/spatial_pooler.py',
};

// Pure hero state for the Spatial Pooler plate. For every column it eases the
// overlap count and each synapse permanence between step idx and idx+1, and
// flags a synapse connected when its eased permanence crosses
// syn_perm_connected. It also lists the connected synapses from the winning
// columns to active input bits (capped), which the hero draws as the reason
// those columns won.
export function spHeroState(run, t, connectedPerm) {
  const N = run.steps.length;
  const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
  const next = Math.min(N - 1, idx + 1);
  const f = ease(t - idx);
  const nP = run.nPotential;
  const n = run.n;
  const cols = [];
  let maxOverlap = 0;
  for (let c = 0; c < n; c++) {
    const overlap = run.steps[idx].overlap[c] * (1 - f) + run.steps[next].overlap[c] * f;
    const winner = run.steps[idx].activeSet.has(c);
    if (overlap > maxOverlap) maxOverlap = overlap;
    const syn = [];
    for (let p = 0; p < nP; p++) {
      const perm = run.steps[idx].perm[c * nP + p] * (1 - f) + run.steps[next].perm[c * nP + p] * f;
      syn.push({ input: run.potential_inputs[c][p], perm, connected: perm >= connectedPerm });
    }
    cols.push({ c, overlap, winner, syn });
  }
  // Connected synapses from winning columns to active input bits (capped).
  const lines = [];
  for (const col of cols) {
    if (!col.winner) continue;
    for (const s of col.syn) {
      if (!s.connected) continue;
      if (!run.input[s.input]) continue;
      lines.push({ inputIdx: s.input, colIdx: col.c });
      if (lines.length >= 80) break;
    }
    if (lines.length >= 80) break;
  }
  const nWin = run.steps[idx].activeSet.size;
  const narration =
    `step ${idx + 1}/${N}: ${nWin} active columns; overlap = connected synapses on active inputs ` +
    `(connected when permanence ≥ ${connectedPerm})`;
  return { idx, f, cols, maxOverlap, input: run.input, lines, narration };
}

// Blend two hex colors; t in [0,1] (t=0 -> a, t=1 -> b).
function mixColor(a, b, t) {
  const pa = hexRgb(a), pb = hexRgb(b);
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function hexRgb(s) {
  const m = String(s).match(/^#([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const inputSizeCtl = ui.number(controls, { key: 'input_size', label: 'input_size', unit: 'bits', value: 64, onInput: () => recompute() });
  // 256 columns with the CoNeX default local_area_density 0.02 -> 5 active
  // columns, so the student sees several winners rather than a single one.
  const nColumnsCtl = ui.number(controls, { key: 'n_columns', label: 'n_columns', unit: 'columns', value: 256, onInput: () => recompute() });
  const potentialPctCtl = ui.slider(controls, { key: 'potential_pct', label: 'potential_pct', unit: 'a.u.', min: 0.1, max: 1, step: 0.05, value: SpatialPoolerDefaults.potential_pct, onInput: () => recompute() });
  const densityCtl = ui.slider(controls, { key: 'local_area_density', label: 'local_area_density', unit: 'a.u.', min: 0.01, max: 0.5, step: 0.01, value: SpatialPoolerDefaults.local_area_density, onInput: () => recompute() });
  const connectedCtl = ui.slider(controls, { key: 'syn_perm_connected', label: 'syn_perm_connected', unit: 'a.u.', min: 0.05, max: 0.5, step: 0.01, value: SpatialPoolerDefaults.syn_perm_connected, onInput: () => recompute() });
  const incCtl = ui.slider(controls, { key: 'syn_perm_active_inc', label: 'syn_perm_active_inc', unit: 'a.u.', min: 0.01, max: 0.2, step: 0.01, value: SpatialPoolerDefaults.syn_perm_active_inc, onInput: () => recompute() });
  const decCtl = ui.slider(controls, { key: 'syn_perm_inactive_dec', label: 'syn_perm_inactive_dec', unit: 'a.u.', min: 0.001, max: 0.05, step: 0.001, value: SpatialPoolerDefaults.syn_perm_inactive_dec, onInput: () => recompute() });
  const boostCtl = ui.slider(controls, { key: 'boost_strength', label: 'boost_strength', unit: 'a.u.', min: 0, max: 2, step: 0.1, value: SpatialPoolerDefaults.boost_strength, onInput: () => recompute() });
  const seedCtl = ui.number(controls, { key: 'seed', label: 'seed', unit: '', value: 7, onInput: () => recompute() });
  const inputBitsCtl = ui.slider(controls, { key: 'input_bits', label: 'active input bits', unit: 'bits', min: 4, max: 32, step: 1, value: 16, onInput: () => recompute() });
  const stepsCtl = ui.number(controls, { key: 'steps', label: 'learning steps', unit: 'steps', value: 20, onInput: () => recompute() });

  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => recompute() });

  // Hero: input bits (left), columns (right) and the connected synapses as
  // thin lines. Fixed size (HERO_COLS), so it is created once, not rebuilt.
  const hero = ui.hero(figures, { height: 240 });
  const heroCap = { set(text) { hero.narration(text); } };

  // Figures are rebuilt on recompute because their size depends on n_columns.
  const figBox = document.createElement('div');
  figures.appendChild(figBox);
  let permHeat = null;
  let usagePlot = null;
  let inputCanvas = null;
  let activeCanvas = null;

  let run = null; // precomputed learning run
  let lastT = 0; // current player time, in steps

  // The player replays the precomputed steps one by one. integer:false so
  // onFrame gets the fractional time and the hero eases between steps.
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'steps', playSeconds: 5, loop: true, integer: false, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const N = run.steps.length;
    const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
    const st = run.steps[idx];

    // Permanence heatmap at this step (row-major, one row per column).
    permHeat.draw(st.perm, { min: 0, max: 1 });

    // Column usage so far: fraction of steps each column won.
    const usage = new Array(run.n).fill(0);
    for (let s = 0; s <= idx; s++) for (const c of run.steps[s].activeSet) usage[c]++;
    usagePlot.draw([{ x: usage.map((_, c) => c), y: usage.map((u) => u / (idx + 1)), label: 'usage', color: 'c1', style: 'bars' }], {
      xRange: [0, run.n],
      yRange: [0, 1],
    });

    drawActive(idx);
    drawHero();
    updateNotes(idx);
  }

  // Hero: the input as a square grid of cells (left), the columns as a square
  // grid (right) whose fill is the eased overlap, and the connected synapses
  // from the winning columns to active input bits as thin teal curves between.
  function drawHero() {
    const st = spHeroState(run, lastT, connectedCtl.get());
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const pad = 8;
    const nIn = run.input.length;
    const nCols = run.n;
    const inSide = Math.floor(Math.sqrt(nIn));
    const colSide = Math.floor(Math.sqrt(nCols));
    const gap = 2;
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c5 = ui.color('c5');
    const muted = ui.color('muted');
    const paper = ui.color('paper');

    // Side by side when wide; stacked (input on top, columns below) when narrow
    // so the two grids never overlap.
    const stacked = width < 600;
    let inX, inY, inCell, colX, colY, colCell;
    if (stacked) {
      const midGap = 24;
      const gridSide = Math.min(width - 2 * pad, (height - 2 * pad - midGap) / 2);
      inCell = (gridSide - gap * (inSide - 1)) / inSide;
      colCell = (gridSide - gap * (colSide - 1)) / colSide;
      inX = (width - gridSide) / 2; inY = pad;
      colX = (width - gridSide) / 2; colY = pad + gridSide + midGap;
    } else {
      const gridH = height - 2 * pad;
      inCell = (gridH - gap * (inSide - 1)) / inSide;
      colCell = (gridH - gap * (colSide - 1)) / colSide;
      inX = pad; inY = pad;
      colX = width - pad - colSide * (colCell + gap) + gap; colY = pad;
    }

    // Input grid: active bits filled green, inactive faint.
    for (let i = 0; i < nIn; i++) {
      const r = Math.floor(i / inSide);
      const c = i % inSide;
      const x = inX + c * (inCell + gap);
      const y = inY + r * (inCell + gap);
      ctx.fillStyle = run.input[i] ? c5 : muted;
      ctx.globalAlpha = run.input[i] ? 1 : 0.25;
      ctx.fillRect(x, y, inCell, inCell);
      ctx.globalAlpha = 1;
    }

    // Column grid: fill = eased overlap / max overlap (teal ramp);
    // winners get a coral ring.
    const maxOv = Math.max(1, st.maxOverlap);
    for (const col of st.cols) {
      const r = Math.floor(col.c / colSide);
      const c = col.c % colSide;
      const x = colX + c * (colCell + gap);
      const y = colY + r * (colCell + gap);
      const frac = Math.max(0, Math.min(1, col.overlap / maxOv));
      ctx.fillStyle = mixColor(paper, c1, frac);
      ctx.fillRect(x, y, colCell, colCell);
      if (col.winner) {
        ctx.strokeStyle = c2;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, colCell - 2, colCell - 2);
      }
    }

    // Connected synapses from winning columns to active input bits (teal curves).
    ctx.strokeStyle = c1;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.4;
    for (const ln of st.lines) {
      const ir = Math.floor(ln.inputIdx / inSide);
      const ic = ln.inputIdx % inSide;
      const cr = Math.floor(ln.colIdx / colSide);
      const cc = ln.colIdx % colSide;
      let x0, y0, x1, y1;
      if (stacked) {
        // from the bottom of the input cell to the top of the column cell
        x0 = inX + ic * (inCell + gap) + inCell / 2;
        y0 = inY + ir * (inCell + gap) + inCell;
        x1 = colX + cc * (colCell + gap) + colCell / 2;
        y1 = colY + cr * (colCell + gap);
        const my = (y0 + y1) / 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.quadraticCurveTo(x0, my, (x0 + x1) / 2, my);
        ctx.quadraticCurveTo(x1, my, x1, y1);
        ctx.stroke();
      } else {
        // from the right of the input cell to the left of the column cell
        x0 = inX + ic * (inCell + gap) + inCell;
        y0 = inY + ir * (inCell + gap) + inCell / 2;
        x1 = colX + cc * (colCell + gap);
        y1 = colY + cr * (colCell + gap) + colCell / 2;
        const mx = (x0 + x1) / 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.quadraticCurveTo(mx, y0, mx, (y0 + y1) / 2);
        ctx.quadraticCurveTo(mx, y1, x1, y1);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    heroCap.set(st.narration);
  }
  hero.onResize(() => { if (run) drawHero(); });

  // Input bits (fixed for the whole run) and the winning columns at a step.
  function drawInput() {
    const { ctx, width, height } = inputCanvas;
    ctx.clearRect(0, 0, width, height);
    const n = run.input.length;
    const pad = 4;
    const gap = 1;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = run.input[i] ? ui.color('c5') : ui.color('muted');
      ctx.fillRect(pad + i * (bw + gap), pad, Math.max(1, bw - 1), height - 2 * pad);
    }
  }
  function drawActive(idx) {
    const { ctx, width, height } = activeCanvas;
    ctx.clearRect(0, 0, width, height);
    const n = run.n;
    const pad = 4;
    const bw = (width - 2 * pad) / n;
    const set = run.steps[idx].activeSet;
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = set.has(c) ? ui.color('c2') : ui.color('muted');
      ctx.fillRect(pad + c * bw, pad, Math.max(1, bw - 1), height - 2 * pad);
    }
  }

  // ui.canvas does not expose its <figure>, and site.css is shell-owned, so the
  // strip captions are styled inline to match the .ui-figure figcaption look.
  function caption(text) {
    const d = document.createElement('div');
    d.style.cssText = 'font-style:italic;color:var(--muted);font-size:0.85rem;margin-top:0.3rem;';
    d.textContent = text;
    figBox.appendChild(d);
    return d;
  }

  function rebuildFigures() {
    figBox.innerHTML = '';
    permHeat = ui.heatmap(figBox, {
      rows: run.n,
      cols: run.nPotential,
      label: `permanence 0–1 a.u. (rows=columns, cols=potential inputs); connected ≥ syn_perm_connected = ${connectedCtl.get()}`,
    });
    usagePlot = ui.plot(figBox, { height: 200, xLabel: 'column', yLabel: 'usage', xUnit: 'index', yUnit: 'a.u.' });
    inputCanvas = ui.canvas(figBox, { height: 60 });
    caption('input bits at this step (green = active)');
    activeCanvas = ui.canvas(figBox, { height: 60 });
    caption('columns (coral = winners this step)');
    // The canvas backing store is reset on layout/resize, so register the
    // draws via onResize (called after layout and on every resize).
    if (typeof inputCanvas.onResize === 'function') inputCanvas.onResize(drawInput);
    if (typeof activeCanvas.onResize === 'function') activeCanvas.onResize(() => drawActive(Math.max(0, Math.min(run.steps.length - 1, Math.floor(lastT)))));
  }

  function recompute() {
    const cfg = {
      input_size: inputSizeCtl.get(),
      n_columns: nColumnsCtl.get(),
      potential_pct: potentialPctCtl.get(),
      local_area_density: densityCtl.get(),
      syn_perm_connected: connectedCtl.get(),
      syn_perm_active_inc: incCtl.get(),
      syn_perm_inactive_dec: decCtl.get(),
      boost_strength: boostCtl.get(),
    };
    const sp = new SpatialPooler(cfg);
    const { n_columns, n_potential, input_size } = { n_columns: sp.n_columns, n_potential: sp.n_potential, input_size: sp.cfg.input_size };
    const rng = ui.rng(seedCtl.get());
    // Random potential map: each column samples n_potential distinct inputs.
    sp.potential_inputs = Array.from({ length: n_columns }, () => {
      const s = new Set();
      while (s.size < n_potential) s.add(Math.floor(rng.next() * input_size));
      return [...s];
    });
    // Initial permanences: rand*0.2 + connected - 0.1 (CoNeX init).
    sp.permanences = Array.from({ length: n_columns }, () =>
      Array.from({ length: n_potential }, () => rng.next() * 0.2 + connectedCtl.get() - 0.1)
    );
    sp.boost_factors = new Array(n_columns).fill(1.0);
    sp.overlap_duty = new Array(n_columns).fill(0.0);
    sp.active_duty = new Array(n_columns).fill(0.0);

    const k = Math.min(inputBitsCtl.get(), input_size);
    const rngIn = ui.rng(seedCtl.get() + 101);
    const idx = new Set();
    while (idx.size < k) idx.add(Math.floor(rngIn.next() * input_size));
    const input = new Array(input_size).fill(0);
    for (const i of idx) input[i] = 1;

    // Run all learning steps up front; the player only indexes into this.
    const N = Math.max(1, stepsCtl.get());
    const steps = [];
    for (let s = 0; s < N; s++) {
      const active = sp.compute(input, true);
      steps.push({
        activeSet: new Set(active),
        overlap: sp.overlap.slice(),
        perm: sp.permanences.flat(),
      });
    }
    run = { n: n_columns, nPotential: n_potential, input, steps, potential_inputs: sp.potential_inputs };
    rebuildFigures();
    drawInput();
    if (player) {
      player.setDuration(N);
      // Keep the current position (clamped) so a parameter change does not restart the run.
      player.seek(Math.min(lastT, N));
    } else {
      frame(N); // no player available: show the finished run
    }
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student steps through.
  let dynNotes = null;

  function updateNotes(idx) {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const st = run.steps[idx];
    const winners = [...st.activeSet].sort((a, b) => a - b);
    const winOverlap = winners.map((c) => st.overlap[c]);
    const nActive = Math.max(1, Math.trunc(run.n * densityCtl.get()));
    ui.note(
      dynNotes,
      `<p>The Spatial Pooler turns an input pattern into a fixed-size SDR of <b>${nActive}</b> active columns out of <b>${run.n}</b> ` +
        `(local_area_density = ${densityCtl.get()}). Each column has a pool of potential inputs; a synapse is <i>connected</i> when its ` +
        `permanence ≥ <b>syn_perm_connected = ${connectedCtl.get()}</b>. Overlap counts connected synapses on active inputs; the top-k ` +
        `columns win (global inhibition). Step <b>${idx + 1}/${run.steps.length}</b>: winners <b>[${winners.join(', ')}]</b> with overlaps ` +
        `<b>[${winOverlap.join(', ')}]</b> bits. ` +
        `Note: in the current CoNeX code the <code>global_inhibition=False</code> branch runs the same top-k as the global branch ` +
        `(<code>spatial_pooler.py</code> <code>_inhibit</code>), so a "local inhibition" toggle would give the same result.</p>` +
        `<p>Learning raises permanence on active inputs by <b>${incCtl.get()}</b> and lowers it on inactive ones by <b>${decCtl.get()}</b>, ` +
        `clipped to [0,1]. With <b>boost_strength = ${boostCtl.get()}</b>, columns that fire less than the target duty cycle get a boost ` +
        `factor &gt; 1 so every column eventually participates (homeostasis). All permanences and densities are dimensionless (a.u.).</p>`
    );
    ui.equation(
      dynNotes,
      `<div>overlap<sub>c</sub> = <var>Σ</var><sub>p</sub> [perm<sub>cp</sub> ≥ conn] · [input<sub>p</sub> &gt; 0], &nbsp; ` +
        `boost<sub>c</sub> = exp(<var>β</var>·(target − duty<sub>c</sub>))</div>`
    );
    ui.note(
      dynNotes,
      `<p><b>Background.</b> The Spatial Pooler is the input layer of HTM: it sparsifies and stabilizes sensory input so similar inputs ` +
        `produce similar column SDRs. <b>ML analogy:</b> competitive / sparse coding (like a winner-take-all autoencoder). ` +
        `The analogy breaks because permanence learning is local and homeostatic, not a global loss gradient.</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from pymonntorch import Network, NeuronGroup\n` +
      `from conex import SpatialPooler, SpatialPoolerConfig, TimeResolution\n` +
      `cfg = SpatialPoolerConfig(input_size=${inputSizeCtl.get()}, n_columns=${run.n},\n` +
      `    potential_pct=${potentialPctCtl.get()}, local_area_density=${densityCtl.get()},\n` +
      `    syn_perm_connected=${connectedCtl.get()}, syn_perm_active_inc=${incCtl.get()},\n` +
      `    syn_perm_inactive_dec=${decCtl.get()}, boost_strength=${boostCtl.get()}, seed=${seedCtl.get()})\n` +
      `net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")\n` +
      `neurons = NeuronGroup(net=net, size=${run.n}, behavior={100: SpatialPooler(config=cfg)}, tag="sp")\n` +
      `net.initialize()\n` +
      `sp = neurons.spatial_pooler\n` +
      `x = torch.zeros(${inputSizeCtl.get()}); x[8:16] = 1.0\n` +
      `active = sp.compute(neurons, x, learn=True)`
    );
  }

  function staticNotes() {
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}
