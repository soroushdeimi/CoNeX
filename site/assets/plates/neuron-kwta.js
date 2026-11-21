import { DEFAULTS, lifUpdate, kwta, inherentNoise, kwtaNarration } from '../js/models/neurons.js';

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

  function params() {
    return { R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get() };
  }

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

  // Run the group for STEPS steps. Both runs apply the same seeded noise; the only
  // difference is whether KWTA resets the losers before the spike decision.
  // Returns {spikes: [t, index][], counts: {t, n}[], voltages: [v[] per step],
  // winners: [Set of winner indices per step], nAbove: [count above threshold]}.
  function simRun(useKwta) {
    const p = params();
    const rng = ui.rng(cSeed.get());
    const I = cI.get();
    const scale = cScale.get();
    const offset = cOffset.get();
    const k = cK.get();
    let vs = new Array(SIZE).fill(p.v_rest);
    const spikes = [];
    const counts = [];
    const voltages = [];
    const winners = [];
    const nAbove = [];
    for (let it = 1; it <= STEPS; it++) {
      const t = it * p.dt;
      vs = vs.map((v) => lifUpdate({ v }, p, I, p.dt));
      vs = inherentNoise(vs, rng, scale, offset);
      const above = vs.filter((v) => v >= p.threshold).length;
      nAbove.push(above);
      let win = new Set();
      if (useKwta) {
        const after = kwta(vs, p.threshold, p.v_reset, k);
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
    return { spikes, counts, voltages, winners, nAbove };
  }

  // --- figures ---
  const hero = ui.hero(figures, { height: 300 });
  const figBefore = ui.raster(figures, { height: 180, xLabel: 't', xUnit: 'ms' });
  const figAfter = ui.raster(figures, { height: 180, xLabel: 't', xUnit: 'ms' });
  const figCount = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'spiking neurons', xUnit: 'ms', yUnit: 'count' });

  // --- notes ---
  ui.note(notes, `KWTA keeps only the <var>k</var> strongest neurons active. In CoNeX, <code>KWTA.forward</code>
  checks which neurons are above threshold; if more than <var>k</var> are, it keeps the top-<var>k</var> by voltage and
  resets every other spiking neuron to <var>v_reset</var>. If at most <var>k</var> are above threshold, nothing changes.
  <code>InherentNoise</code> adds a uniform random voltage offset (mV) each step so the group's activity varies.
  Units: v in mV, t in ms, I in pA, R in GOhm. The hero shows the group as circles: the fill is each neuron's
  voltage, the <var>k</var> winners glow, and the losers dim with an inhibition ripple when they are reset.`);
  ui.note(notes, `The two rasters use the <b>same seeded noise</b>: the top is the group with no KWTA
  (every neuron that crosses threshold fires), the bottom is the same group with KWTA applied, so the
  difference is exactly the k-winners inhibition. The bottom plot shows the number of spiking neurons
  per step for both runs: without KWTA it can exceed <var>k</var>, with KWTA it is capped at <var>k</var>
  (the dashed line). Raising <var>k</var> lets more neurons win; lowering it sharpens the competition.
  ML analogy: KWTA is a hard top-<var>k</var> sparsification, like keeping only the <var>k</var> largest
  activations; the analogy breaks because here the "activations" are membrane voltages with dynamics,
  not a static feature vector.`);
  ui.code(notes, () => codeSnippet());

  // --- hero: neurons as circles, fill = voltage, winners glow, losers dim ---
  let lastReal = 0;
  function drawHero(t) {
    const { p, after } = runData;
    const k = cK.get();
    // interpolate between step k and k+1 with the fractional part (AMENDMENT 4.3)
    const stepF = Math.min(after.voltages.length - 1, Math.max(0, t / p.dt));
    const k0 = Math.floor(stepF);
    const f = ui.ease(stepF - k0);
    const v0 = after.voltages[k0];
    const v1 = after.voltages[Math.min(k0 + 1, after.voltages.length - 1)];
    const vs = v0.map((v, j) => v * (1 - f) + v1[j] * f);
    const win = after.winners[k0];
    const nAbove = after.nAbove[k0];

    const ctx = hero.ctx;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), coral: ui.color('c2'), indigo: ui.color('c4'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const cols = 4, rows = 2;
    const gx0 = W * 0.08, gy0 = H * 0.16;
    const cellW = (W * 0.84) / cols, cellH = (H * 0.66) / rows;
    const rad = Math.min(cellW, cellH) * 0.32;
    const vLo = Math.min(p.v_rest, p.v_reset) - 5;
    const vHi = p.threshold + 5;
    for (let j = 0; j < SIZE; j++) {
      const cx = gx0 + (j % cols) * cellW + cellW / 2;
      const cy = gy0 + Math.floor(j / cols) * cellH + cellH / 2;
      const frac = Math.max(0, Math.min(1, (vs[j] - vLo) / (vHi - vLo)));
      const isWinner = win.has(j);
      const isLoser = vs[j] >= p.threshold && !isWinner; // above threshold but reset
      // fill = voltage (teal); winners glow coral, losers dim indigo
      ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      if (isWinner) {
        ctx.fillStyle = pal.coral; ctx.globalAlpha = 0.5 + 0.4 * frac; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = pal.coral; ctx.lineWidth = 3;
      } else if (isLoser) {
        ctx.fillStyle = pal.indigo; ctx.globalAlpha = 0.35; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = pal.indigo; ctx.lineWidth = 2;
      } else {
        ctx.fillStyle = pal.water; ctx.globalAlpha = 0.2 + 0.5 * frac; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = pal.ink; ctx.lineWidth = 1.5;
      }
      ctx.stroke();
      // inhibition ripple on a loser (a ring just outside the circle)
      if (isLoser) {
        ctx.strokeStyle = pal.indigo; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, rad + 5, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(String(j), cx, cy + rad + 6);
    }
    ctx.fillStyle = pal.ink; ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`fill = v (mV); coral = ${k} winners, indigo = reset losers`, gx0, H * 0.06);
  }

  // --- frame: reveal the rasters and count plot, draw the hero ---
  let curT = 0;
  function frame(t) {
    if (!runData) return;
    curT = t;
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
    drawHero(t);
    const step = Math.min(after.nAbove.length - 1, Math.max(0, Math.round(t / p.dt)));
    hero.narration(kwtaNarration({ nAbove: after.nAbove[step], k }));
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

  // On a parameter change: recompute both runs and continue from the current t (clamped).
  function run() {
    player.setDuration(computeRun());
  }

  computeRun();
  const player = ui.player(figures, { duration: runData.tMax, unit: 'ms', playSeconds: 5, loop: true, onFrame: frame });
  frame(0);
  hero.onResize(() => frame(curT));

  const all = [cR, cTau, cTh, cReset, cRest, cDt, cSeed, cI, cScale, cOffset, cK];
  for (const c of all) c.onInput(run);

  return { destroy() {} };
}
