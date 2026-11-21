import { DEFAULTS, lifUpdate, kwta, inherentNoise } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-kwta',
  title: 'k-winners-take-all',
  conex: ['KWTA', 'InherentNoise'],
  source: 'conex/behaviors/neurons/specs.py',
};

const P = { ...DEFAULTS };
const SIZE = 8;
const STEPS = 200;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Neuron (LIF) parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });

  const secIn = ui.section(controls, 'Input and noise (seeded)');
  const cSeed = ui.number(secIn, { key: 'seed', label: 'seed', unit: '', value: 0 });
  const cI = ui.slider(secIn, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });
  const cScale = ui.slider(secIn, { key: 'scale', label: 'noise scale', unit: 'mV', min: 0, max: 20, step: 0.5, value: 5.0 });
  const cOffset = ui.slider(secIn, { key: 'offset', label: 'noise offset', unit: 'mV', min: -10, max: 10, step: 0.5, value: 0.0 });

  const secK = ui.section(controls, 'KWTA');
  const cK = ui.slider(secK, { key: 'k', label: 'k (winners)', unit: '', min: 1, max: SIZE, step: 1, value: 2 });

  // Compute both runs once; the player reveals them frame by frame.
  let runData = null;
  function computeRun() {
    const p = params();
    const tMax = STEPS * p.dt;
    const before = simRun(false);
    const after = simRun(true);
    runData = { p, tMax, before, after };
    return tMax;
  }

  const figBefore = ui.raster(figures, { height: 200, xLabel: 't', xUnit: 'ms' });
  const figAfter = ui.raster(figures, { height: 200, xLabel: 't', xUnit: 'ms' });
  const figCount = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'spiking neurons', xUnit: 'ms', yUnit: 'count' });
  // Bar chart of the group's voltages at the current step; the k winners are highlighted.
  const figWin = ui.canvas(figures, { height: 160 });
  // Build the run before the player: ui.player may call onFrame synchronously.
  const player = ui.player(figures, { duration: computeRun(), unit: 'ms', playSeconds: 5, loop: true, onFrame: frame });

  ui.note(notes, `KWTA keeps only the <var>k</var> strongest neurons active. In CoNeX, <code>KWTA.forward</code>
  checks which neurons are above threshold; if more than <var>k</var> are, it keeps the top-<var>k</var> by voltage and
  resets every other spiking neuron to <var>v_reset</var>. If at most <var>k</var> are above threshold, nothing changes.
  <code>InherentNoise</code> adds a uniform random voltage offset (mV) each step so the group's activity varies.
  Units: v in mV, t in ms, I in pA, R in GOhm.`);
  ui.note(notes, `The two rasters use the <b>same seeded noise</b>: the top is the group with no KWTA
  (every neuron that crosses threshold fires), the bottom is the same group with KWTA applied, so the
  difference is exactly the k-winners inhibition. The bottom plot shows the number of spiking neurons
  per step for both runs: without KWTA it can exceed <var>k</var>, with KWTA it is capped at <var>k</var>
  (the dashed line). Raising <var>k</var> lets more neurons win; lowering it sharpens the competition.
  ML analogy: KWTA is a hard top-<var>k</var> sparsification, like keeping only the <var>k</var> largest
  activations; the analogy breaks because here the "activations" are membrane voltages with dynamics,
  not a static feature vector.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return { R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get() };
  }

  // Run the group for STEPS steps. Both runs apply the same seeded noise; the only
  // difference is whether KWTA resets the losers before the spike decision.
  // Returns {spikes: [t, index][], counts: {t, n}[], voltages: [v[] per step],
  // winners: [Set of winner indices per step]}. voltages/winners are only filled
  // for the KWTA run (useKwta) and are the post-KWTA voltages before the reset.
  function simRun(useKwta) {
    const p = params();
    const rng = ui.rng(cSeed.get());
    const I = cI.get();
    const scale = cScale.get();
    const offset = cOffset.get();
    const k = cK.get();
    let vs = new Array(SIZE).fill(p.v_rest);
    const spikes = []; // [t, index]
    const counts = []; // {t, n}
    const voltages = []; // v[] per step (post-KWTA, pre-reset)
    const winners = []; // Set of winner indices per step
    for (let it = 1; it <= STEPS; it++) {
      const t = it * p.dt;
      vs = vs.map((v) => lifUpdate({ v }, p, I, p.dt));
      vs = inherentNoise(vs, rng, scale, offset);
      let win = new Set();
      if (useKwta) {
        const after = kwta(vs, p.threshold, p.v_reset, k);
        // Winners are the neurons KWTA left above threshold (the top-k).
        for (let j = 0; j < SIZE; j++) if (after[j] >= p.threshold) win.add(j);
        vs = after;
      }
      voltages.push(vs.slice());
      winners.push(win);
      let n = 0;
      for (let j = 0; j < SIZE; j++) {
        if (vs[j] >= p.threshold) { spikes.push([t, j]); vs[j] = p.v_reset; n++; }
      }
      counts.push({ t, n });
    }
    return { spikes, counts, voltages, winners };
  }

  // On a parameter change: recompute both runs and continue from the current t (clamped).
  function run() {
    player.setDuration(computeRun());
  }

  // Reveal both rasters and the count plot up to time t, and highlight the k
  // winners at the current step.
  function frame(t) {
    if (!runData) return;
    const { p, tMax, before, after } = runData;
    const k = cK.get();
    figBefore.draw(before.spikes, SIZE, tMax, { revealT: t });
    figAfter.draw(after.spikes, SIZE, tMax, { revealT: t });
    figCount.draw(
      [
        { x: before.counts.map((c) => c.t), y: before.counts.map((c) => c.n), label: 'no KWTA', color: 'c3', style: 'line' },
        { x: after.counts.map((c) => c.t), y: after.counts.map((c) => c.n), label: 'KWTA', color: 'c1', style: 'line' },
      ],
      { revealX: t, hlines: [{ y: k, label: 'k' }], yRange: [0, SIZE] }
    );
    drawWinners(t);
  }

  // Bar chart of the group's voltages at the current step; the k winners (the
  // top-k above threshold in the KWTA run) are drawn in coral, the rest in teal.
  function drawWinners(t) {
    const { p, after } = runData;
    const step = Math.min(after.voltages.length - 1, Math.max(0, Math.round(t / p.dt)));
    const vs = after.voltages[step];
    const win = after.winners[step];
    const ctx = figWin.ctx;
    const W = figWin.width, H = figWin.height;
    const M = { left: 44, right: 12, top: 12, bottom: 30 };
    ctx.clearRect(0, 0, W, H);
    const pal = { ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule') };
    const vMin = Math.min(p.v_rest, p.v_reset) - 5;
    const vMax = p.threshold + 5;
    const sy = (v) => (H - M.bottom) - ((v - vMin) / (vMax - vMin)) * (H - M.bottom - M.top);
    // frame + threshold line
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1;
    ctx.strokeRect(M.left, M.top, W - M.left - M.right, H - M.bottom - M.top);
    ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(M.left, sy(p.threshold)); ctx.lineTo(W - M.right, sy(p.threshold)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`threshold ${p.threshold} mV`, M.left + 3, sy(p.threshold) - 12);
    // bars
    const bw = (W - M.left - M.right) / SIZE;
    for (let j = 0; j < SIZE; j++) {
      const x = M.left + j * bw;
      const y = sy(Math.max(vMin, Math.min(vMax, vs[j])));
      ctx.fillStyle = win.has(j) ? ui.color('c2') : ui.color('c1');
      ctx.fillRect(x + 2, Math.min(y, H - M.bottom), bw - 4, Math.abs((H - M.bottom) - y));
      ctx.fillStyle = pal.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(String(j), x + bw / 2, H - M.bottom + 3);
    }
    ctx.fillStyle = pal.ink; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`voltage of each neuron at t = ${(step * p.dt).toFixed(1)} ms (coral = k winners)`, (M.left + W - M.right) / 2, H - 2);
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup, NeuronDimension
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire, KWTA, InherentNoise
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=1, width=${SIZE}), behavior=[
    LIF(R=${p.R}, threshold=${cTh.get()}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    InherentNoise(mode="rand", scale=${cScale.get()}, offset=${cOffset.get()}),
    KWTA(k=${cK.get()}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  const all = [cR, cTau, cTh, cReset, cRest, cDt, cSeed, cI, cScale, cOffset, cK];
  for (const c of all) c.onInput(run);
  ui.onTheme(run);

  return { destroy() {} };
}
