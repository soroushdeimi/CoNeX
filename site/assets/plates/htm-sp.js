import { SpatialPooler, SpatialPoolerDefaults } from '../js/models/htm.js';

export const meta = {
  id: 'htm-sp',
  title: 'Spatial Pooler',
  conex: ['SpatialPooler', 'SpatialPoolerConfig'],
  source: 'conex/behaviors/neurons/numenta/spatial_pooler.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const inputSizeCtl = ui.number(controls, { key: 'input_size', label: 'input_size', unit: 'bits', value: 64, onInput: () => reset() });
  // 256 columns with the CoNeX default local_area_density 0.02 -> 5 active
  // columns, so the student sees several winners rather than a single one.
  const nColumnsCtl = ui.number(controls, { key: 'n_columns', label: 'n_columns', unit: 'columns', value: 256, onInput: () => reset() });
  const potentialPctCtl = ui.slider(controls, { key: 'potential_pct', label: 'potential_pct', unit: 'a.u.', min: 0.1, max: 1, step: 0.05, value: SpatialPoolerDefaults.potential_pct, onInput: () => reset() });
  const densityCtl = ui.slider(controls, { key: 'local_area_density', label: 'local_area_density', unit: 'a.u.', min: 0.01, max: 0.5, step: 0.01, value: SpatialPoolerDefaults.local_area_density, onInput: () => reset() });
  const connectedCtl = ui.slider(controls, { key: 'syn_perm_connected', label: 'syn_perm_connected', unit: 'a.u.', min: 0.05, max: 0.5, step: 0.01, value: SpatialPoolerDefaults.syn_perm_connected, onInput: () => reset() });
  const incCtl = ui.slider(controls, { key: 'syn_perm_active_inc', label: 'syn_perm_active_inc', unit: 'a.u.', min: 0.01, max: 0.2, step: 0.01, value: SpatialPoolerDefaults.syn_perm_active_inc, onInput: () => reset() });
  const decCtl = ui.slider(controls, { key: 'syn_perm_inactive_dec', label: 'syn_perm_inactive_dec', unit: 'a.u.', min: 0.001, max: 0.05, step: 0.001, value: SpatialPoolerDefaults.syn_perm_inactive_dec, onInput: () => reset() });
  const boostCtl = ui.slider(controls, { key: 'boost_strength', label: 'boost_strength', unit: 'a.u.', min: 0, max: 2, step: 0.1, value: SpatialPoolerDefaults.boost_strength, onInput: () => reset() });
  const seedCtl = ui.number(controls, { key: 'seed', label: 'seed', unit: '', value: 7, onInput: () => reset() });
  const inputBitsCtl = ui.slider(controls, { key: 'input_bits', label: 'active input bits', unit: 'bits', min: 4, max: 32, step: 1, value: 16, onInput: () => reset() });

  const stepBtn = ui.button(controls, { label: 'Step (learn)', onClick: () => step(true) });
  const inferBtn = ui.button(controls, { label: 'Step (no learn)', onClick: () => step(false) });
  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => reset() });

  // Figures are rebuilt on reset because their size depends on n_columns.
  const figBox = document.createElement('div');
  figures.appendChild(figBox);
  let permHeat = null;
  let usagePlot = null;
  let activeCanvas = null;

  let sp = null;
  let input = null;
  let history = []; // {step, nActive, activeSet}
  let stepCount = 0;
  let booting = false;

  function rebuildFigures() {
    figBox.innerHTML = '';
    permHeat = ui.heatmap(figBox, {
      rows: sp.n_columns,
      cols: sp.n_potential,
      label: 'permanence (rows=columns, cols=potential inputs)',
    });
    usagePlot = ui.plot(figBox, { height: 200, xLabel: 'column', yLabel: 'usage', xUnit: 'index', yUnit: 'a.u.' });
    activeCanvas = ui.canvas(figBox, { height: 60 });
    activeCanvas.onResize(drawActive);
  }

  function buildInput() {
    const n = inputSizeCtl.get();
    const k = Math.min(inputBitsCtl.get(), n);
    const rng = ui.rng(seedCtl.get() + 101);
    const idx = new Set();
    while (idx.size < k) idx.add(Math.floor(rng.next() * n));
    const x = new Array(n).fill(0);
    for (const i of idx) x[i] = 1;
    return x;
  }

  function reset() {
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
    sp = new SpatialPooler(cfg);
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
    input = buildInput();
    history = [];
    stepCount = 0;
    rebuildFigures();
    // Default learning run so the usage curve and heatmap show real content
    // on load; Step/Play continue from here. booting suppresses the notes
    // until the run finishes so they are not appended 20 times.
    booting = true;
    for (let i = 0; i < 20; i++) step(true);
    booting = false;
    updateNotes();
  }

  function step(learn) {
    if (!sp) return;
    const active = sp.compute(input, learn);
    stepCount += 1;
    history.push({ step: stepCount, nActive: active.length, activeSet: new Set(active) });
    redraw();
  }

  // Active columns this step. The canvas backing store resets on layout/
  // resize, so this is registered via onResize and also called from redraw.
  function drawActive() {
    if (!sp || !activeCanvas) return;
    const { ctx, width, height } = activeCanvas;
    ctx.clearRect(0, 0, width, height);
    const last = history[history.length - 1];
    const n = sp.n_columns;
    const pad = 4;
    const bw = (width - 2 * pad) / n;
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = last && last.activeSet.has(c) ? ui.color('c2') : ui.color('muted');
      ctx.fillRect(pad + c * bw, pad, Math.max(1, bw - 1), height - 2 * pad);
    }
  }

  function redraw() {
    if (!sp) return;
    // Permanence heatmap (row-major, one row per column).
    if (permHeat) {
      const flat = [];
      for (const row of sp.permanences) for (const p of row) flat.push(p);
      permHeat.draw(flat, { min: 0, max: 1 });
    }

    // Column usage over time: fraction of steps each column won.
    if (usagePlot) {
      const n = sp.n_columns;
      const usage = new Array(n).fill(0);
      for (const h of history) for (const c of h.activeSet) usage[c]++;
      const usageSeries = usage.map((u, c) => ({ x: c, y: history.length ? u / history.length : 0 }));
      usagePlot.draw([{ x: usageSeries.map((p) => p.x), y: usageSeries.map((p) => p.y), label: 'usage', color: 'c1', style: 'bars' }], {
        xRange: [0, n],
        yRange: [0, 1],
      });
    }

    // Active columns this step.
    drawActive();

    if (!booting) updateNotes();
  }

  function updateNotes() {
    const n = sp.n_columns;
    notes.innerHTML = '';
    const nActive = sp.n_active;
    const last = history[history.length - 1];
    ui.note(
      notes,
      `<p>The Spatial Pooler turns an input pattern into a fixed-size SDR of <b>${nActive}</b> active columns out of <b>${n}</b> ` +
        `(local_area_density = ${densityCtl.get()}). Each column has a pool of potential inputs; a synapse is <i>connected</i> when its ` +
        `permanence ≥ <b>syn_perm_connected = ${connectedCtl.get()}</b>. Overlap counts connected synapses on active inputs; the top-` +
        `${nActive} columns win (global inhibition). This step: <b>${last ? last.nActive : 0}</b> active columns. ` +
        `Note: in the current CoNeX code the <code>global_inhibition=False</code> branch runs the same top-k as the global branch ` +
        `(<code>spatial_pooler.py</code> <code>_inhibit</code>), so a "local inhibition" toggle would give the same result.</p>` +
        `<p>Learning raises permanence on active inputs by <b>${incCtl.get()}</b> and lowers it on inactive ones by <b>${decCtl.get()}</b>, ` +
        `clipped to [0,1]. With <b>boost_strength = ${boostCtl.get()}</b>, columns that fire less than the target duty cycle get a boost ` +
        `factor &gt; 1 so every column eventually participates (homeostasis). All permanences and densities are dimensionless (a.u.).</p>`
    );
    ui.equation(
      notes,
      `<div>overlap<sub>c</sub> = <var>Σ</var><sub>p</sub> [perm<sub>cp</sub> ≥ conn] · [input<sub>p</sub> &gt; 0], &nbsp; ` +
        `boost<sub>c</sub> = exp(<var>β</var>·(target − duty<sub>c</sub>))</div>`
    );
    ui.note(
      notes,
      `<p><b>Background.</b> The Spatial Pooler is the input layer of HTM: it sparsifies and stabilizes sensory input so similar inputs ` +
        `produce similar column SDRs. <b>ML analogy:</b> competitive / sparse coding (like a winner-take-all autoencoder). ` +
        `The analogy breaks because permanence learning is local and homeostatic, not a global loss gradient.</p>`
    );
    ui.code(notes, () =>
      `from conex import SpatialPooler, SpatialPoolerConfig\n` +
      `cfg = SpatialPoolerConfig(input_size=${inputSizeCtl.get()}, n_columns=${n},\n` +
      `    potential_pct=${potentialPctCtl.get()}, local_area_density=${densityCtl.get()},\n` +
      `    syn_perm_connected=${connectedCtl.get()}, syn_perm_active_inc=${incCtl.get()},\n` +
      `    syn_perm_inactive_dec=${decCtl.get()}, boost_strength=${boostCtl.get()}, seed=${seedCtl.get()})\n` +
      `pooler = SpatialPooler(config=cfg)\n` +
      `active = pooler.compute(neurons, input, learn=True)`
    );
  }

  reset();
  ui.onTheme(redraw);
  return { destroy() {} };
}
