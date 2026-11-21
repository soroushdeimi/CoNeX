import {
  DEFAULTS,
  lifStep,
  makeActivityHomeostasis,
  makeVoltageHomeostasis,
  homeostasisNarration,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-homeostasis',
  title: 'Homeostatic threshold control',
  conex: ['ActivityBaseHomeostasis', 'VoltageBaseHomeostasis'],
  source: 'conex/behaviors/neurons/homeostasis.py',
};

const P = { ...DEFAULTS };
const SIZE = 8;
const DT = 0.1;      // ms, fixed integration step
const WINDOW = 100;  // ms of model time kept on screen
const MODEL_PER_SECOND = 100;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Neuron (LIF) parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold (init)', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const dtNote = document.createElement('p');
  dtNote.style.fontStyle = 'italic';
  dtNote.style.color = 'var(--muted)';
  dtNote.style.fontSize = '0.85rem';
  dtNote.textContent = `dt = ${DT} ms (fixed integration step; the other parameters update live)`;
  sec.appendChild(dtNote);

  const secIn = ui.section(controls, 'Random input (seeded)');
  const cSeed = ui.number(secIn, { key: 'seed', label: 'seed', unit: '', value: 1 });
  const cIbase = ui.slider(secIn, { key: 'I_base', label: 'I_base', unit: 'pA', min: 0, max: 120, step: 1, value: 40.0 });
  const cIamp = ui.slider(secIn, { key: 'I_amp', label: 'I_amp', unit: 'pA', min: 0, max: 80, step: 1, value: 5.0 });

  const secH = ui.section(controls, 'Homeostasis');
  const cType = ui.select(secH, {
    key: 'type', label: 'type',
    options: [
      { value: 'activity', label: 'ActivityBaseHomeostasis' },
      { value: 'voltage', label: 'VoltageBaseHomeostasis' },
    ],
    value: 'activity',
  });
  const cARate = ui.slider(secH, { key: 'activity_rate', label: 'activity_rate', unit: 'spikes/window', min: 1, max: 20, step: 1, value: 1 });
  const cWin = ui.slider(secH, { key: 'window_size', label: 'window_size', unit: 'steps', min: 100, max: 5000, step: 100, value: 1000 });
  const cUpd = ui.slider(secH, { key: 'updating_rate', label: 'updating_rate', unit: 'mV', min: 0, max: 1, step: 0.01, value: 0.2 });
  const cDecay = ui.slider(secH, { key: 'decay_rate', label: 'decay_rate', unit: '', min: 0.5, max: 1, step: 0.01, value: 1.0 });
  const cTarget = ui.slider(secH, { key: 'target_voltage', label: 'target_voltage', unit: 'mV', min: -70, max: -40, step: 0.5, value: -60.0 });
  const cEta = ui.slider(secH, { key: 'eta_ip', label: 'eta_ip', unit: '1/step', min: 0, max: 0.01, step: 0.0005, value: 0.001 });

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), dt: DT, I_base: cIbase.get(), I_amp: cIamp.get(),
    };
  }

  // --- live model state: the real group, stepped one dt at a time ---
  let vs, thresholds, hos, rng, t, iteration;
  const spikeRing = []; // {t, j} for the last WINDOW ms (for the rate meter + raster)
  let rateRing = [];    // {t, rate} rolling group rate (Hz)
  // Range of the control variable (threshold for activity, v for voltage) across
  // all neurons and time, so the per-neuron ticks are spread over the real range
  // instead of all sitting at the same height.
  let ctrlMin = 0, ctrlMax = 1;

  function buildState() {
    const p = params();
    const type = cType.get();
    vs = new Array(SIZE).fill(p.v_rest);
    thresholds = new Array(SIZE).fill(p.threshold);
    hos = type === 'activity'
      ? Array.from({ length: SIZE }, () => makeActivityHomeostasis({
          activity_rate: cARate.get(), window_size: cWin.get(),
          updating_rate: cUpd.get(), decay_rate: cDecay.get(),
        }))
      : Array.from({ length: SIZE }, () => makeVoltageHomeostasis({
          target_voltage: cTarget.get(), eta_ip: cEta.get(),
        }));
    rng = ui.rng(cSeed.get());
    t = 0;
    iteration = 0;
    spikeRing.length = 0;
    rateRing.length = 0;
    const init = type === 'activity' ? p.threshold : p.v_rest;
    ctrlMin = init;
    ctrlMax = init;
  }

  function step() {
    const p = params();
    const type = cType.get();
    iteration += 1;
    t += DT;
    let groupSpikes = 0;
    for (let j = 0; j < SIZE; j++) {
      const I = p.I_base + rng.next() * p.I_amp;
      const pj = type === 'activity' ? { ...p, threshold: thresholds[j] } : p;
      const r = lifStep({ v: vs[j] }, pj, I, DT);
      vs[j] = r.v;
      if (r.spike) { groupSpikes++; spikeRing.push({ t, j }); }
      if (type === 'activity') {
        thresholds[j] = hos[j].step(r.spike, thresholds[j], iteration);
      } else {
        vs[j] = hos[j].step(vs[j]);
      }
      const cv = type === 'activity' ? thresholds[j] : vs[j];
      if (cv < ctrlMin) ctrlMin = cv;
      if (cv > ctrlMax) ctrlMax = cv;
    }
    // rolling group rate over the last WINDOW ms
    while (spikeRing.length && spikeRing[0].t < t - WINDOW) spikeRing.shift();
    const rate = spikeRing.length / (SIZE * (WINDOW / 1000)); // Hz
    rateRing.push({ t, rate });
    while (rateRing.length && rateRing[0].t < t - WINDOW) rateRing.shift();
  }

  function reset() {
    buildState();
  }

  // --- figures ---
  ui.tryThis(figures, 'Try this: raise <var>I</var><sub>base</sub>; watch the rate climb above target and the thresholds rise to pull it back.');
  const hero = ui.hero(figures, { height: 320 });
  const figRate = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'firing rate', xUnit: 'ms', yUnit: 'Hz' });
  const figRaster = ui.raster(figures, { height: 160, xLabel: 't', xUnit: 'ms' });

  // --- notes ---
  ui.note(notes, `Homeostasis keeps a population near a target. <b>ActivityBaseHomeostasis</b> counts spikes in
  a sliding window and shifts each neuron's threshold up or down so its activity tracks the target
  <var>activity_rate</var>. <b>VoltageBaseHomeostasis</b> instead pushes each neuron's voltage toward a target band
  by accumulating an "exhaustion" term. Both run on top of the same LIF dynamics under a random input.
  Units: v in mV, t in ms, I in pA, R in GOhm. The hero shows the group as circles that flash on a spike,
  each with its own moving threshold marker, and a live rate meter against the target.`);
  ui.equation(notes, `activity: <var>threshold</var> ← <var>threshold</var> + <var>activities</var>·<var>updating_rate</var> (every <var>window_size</var> steps)<br>
  voltage: <var>exhaustion</var> += (<var>v</var>−band)·<var>eta_ip</var>, &nbsp; <var>v</var> ← <var>v</var> − <var>exhaustion</var>`);
  ui.note(notes, `The rate meter shows the group firing rate (Hz) over the last 100 ms; the dashed line is the
  target rate. The circles flash coral on a spike; the small tick on each circle is that neuron's current
  threshold (activity-based) or its voltage (voltage-based), so you can watch the control variable move
  to bring the rate to the target. The defaults aim for a cortical rate (~10 Hz): 1 spike per 100 ms
  window. The input (I_base 40 pA, I_amp 5 pA) is chosen so the starting rate is about 2× the target
  (~20 Hz); with R at the neuron default (0.5 GOhm) the threshold then settles within a couple of mV of
  the initial −50 mV, so it stays physiological. ML analogy: this is like a learning-rate / weight-decay
  loop that keeps activation levels in a healthy range; the analogy breaks because it is a slow, local,
  activity-driven rule with no error signal or label.`);
  ui.code(notes, () => codeSnippet());

  // --- hero drawing: neuron circles + rate meter ---
  function drawHero() {
    const p = params();
    const type = cType.get();
    const ctx = hero.ctx;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), coral: ui.color('c2'), ochre: ui.color('c3'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';

    // ---- left: the group of neurons as circles ----
    const cols = 4, rows = 2;
    const gx0 = W * 0.06, gy0 = H * 0.18;
    const cellW = (W * 0.52) / cols, cellH = (H * 0.6) / rows;
    const rad = Math.min(cellW, cellH) * 0.30;
    // v range for the fill: v_rest .. threshold (init)
    const vLo = p.v_rest, vHi = Math.max(p.threshold, p.v_reset) + 5;
    // control-variable range across the run (so the ticks spread out)
    const cLo = ctrlMin, cHi = ctrlMax + (ctrlMax - ctrlMin < 1e-6 ? 1 : 0);
    for (let j = 0; j < SIZE; j++) {
      const cx = gx0 + (j % cols) * cellW + cellW / 2;
      const cy = gy0 + Math.floor(j / cols) * cellH + cellH / 2;
      // flash if this neuron spiked in the last 0.5 ms of model time
      const flashed = spikeRing.some((s) => s.j === j && s.t > t - 0.5);
      // fill level = v mapped to the circle (teal), flash coral on a spike
      const frac = Math.max(0, Math.min(1, (vs[j] - vLo) / (vHi - vLo)));
      ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      ctx.fillStyle = pal.water; ctx.globalAlpha = 0.25 + 0.55 * frac; ctx.fill();
      ctx.globalAlpha = 1;
      // coral glow that decays over ~100 ms of real time after a spike
      if (flashed) {
        ctx.strokeStyle = pal.coral; ctx.lineWidth = 3;
        ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.arc(cx, cy, rad + 4, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = flashed ? pal.coral : pal.ink;
      ctx.lineWidth = flashed ? 3 : 1.5;
      ctx.stroke();
      // control-variable marker: a tick on the circle at the neuron's threshold
      // (activity) or voltage (voltage), mapped over the real run range
      const th = type === 'activity' ? thresholds[j] : vs[j];
      const thFrac = Math.max(0, Math.min(1, (th - cLo) / (cHi - cLo)));
      const ty = cy + rad - thFrac * 2 * rad;
      ctx.strokeStyle = pal.ochre; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx - rad - 4, ty); ctx.lineTo(cx + rad + 4, ty); ctx.stroke();
      // print the value for one neuron (j=0) so the scale is readable; placed
      // to the right of the tick row, clear of the tick line itself
      if (j === 0) {
        ctx.fillStyle = pal.ochre; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`${th.toFixed(1)} mV`, cx + rad + 12, ty - 3);
      }
      ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(String(j), cx, cy + rad + 4);
    }
    ctx.fillStyle = pal.muted; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(type === 'activity' ? 'neurons (tick = threshold)' : 'neurons (tick = v)', gx0, H * 0.04);

    // ---- right: live rate meter with the target line ----
    const target = type === 'activity'
      ? cARate.get() / (cWin.get() * DT / 1000)
      : 0;
    const mx0 = W * 0.64, mx1 = W * 0.96, my0 = H * 0.18, my1 = H * 0.82;
    const curRate = rateRing.length ? rateRing[rateRing.length - 1].rate : 0;
    const maxRate = Math.max(target * 3, 50);
    const yOf = (r) => my1 - (Math.max(0, Math.min(maxRate, r)) / maxRate) * (my1 - my0);
    // frame
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1;
    ctx.strokeRect(mx0, my0, mx1 - mx0, my1 - my0);
    // y tick labels (Hz)
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const rateStep = maxRate > 100 ? 50 : 10;
    for (let r = 0; r <= maxRate; r += rateStep) {
      ctx.fillText(String(r), mx0 - 4, yOf(r));
    }
    // target line
    if (type === 'activity') {
      ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(mx0, yOf(target)); ctx.lineTo(mx1, yOf(target)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted; ctx.font = mono;
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`target ${target.toFixed(0)} Hz`, mx0 + 2, yOf(target) - 2);
    }
    // rate trace
    ctx.strokeStyle = pal.water; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k < rateRing.length; k++) {
      const px = mx0 + ((rateRing[k].t - (t - WINDOW)) / WINDOW) * (mx1 - mx0);
      const py = yOf(rateRing[k].rate);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // current rate readout
    ctx.fillStyle = pal.ink; ctx.font = mono;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`rate ${curRate.toFixed(0)} Hz`, (mx0 + mx1) / 2, my0 - 4);
  }

  // --- frame ---
  function frame() {
    const type = cType.get();
    const target = type === 'activity' ? cARate.get() / (cWin.get() * DT / 1000) : 0;
    const curRate = rateRing.length ? rateRing[rateRing.length - 1].rate : 0;
    drawHero();
    const xs = rateRing.map((r) => r.t);
    const ys = rateRing.map((r) => r.rate);
    figRate.draw([{ x: xs, y: ys, label: 'firing rate', color: 'c1', style: 'line' }],
      { xRange: [t - WINDOW, t], hlines: type === 'activity' ? [{ y: target, label: 'target' }] : [], yRange: [0, Math.max(target * 3, 50)] });
    const spikes = spikeRing.map((s) => [s.t, s.j]);
    figRaster.draw(spikes, SIZE, t, { xRange: [t - WINDOW, t] });
    hero.narration(homeostasisNarration({ rate: curRate, target, mode: type }));
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.neurons.homeostasis import ActivityBaseHomeostasis, VoltageBaseHomeostasis
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${DT})])
pop = NeuronGroup(net=net, size=${SIZE}, behavior=[
    LIF(R=${p.R}, threshold=${cTh.get()}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    Fire(),
    ActivityBaseHomeostasis(activity_rate=${cARate.get()}, window_size=${cWin.get()},
                            updating_rate=${cUpd.get()}, decay_rate=${cDecay.get()}),
    # or: VoltageBaseHomeostasis(target_voltage=${cTarget.get()}, eta_ip=${cEta.get()}),
])
net.initialize()
pop.threshold = pop.vector(mode=${cTh.get()})
pop.I = pop.vector(mode=${p.I_base})   # constant input current (pA)
net.simulate_iterations(10000)`;
  }

  // Changing the mode or a size/seed control rebuilds the state; the other
  // parameters apply on the next step without a reset.
  function rebuild() {
    buildState();
    frame();
  }
  function live() {
    frame();
  }

  buildState();
  ui.live(figures, {
    dt: DT, unit: 'ms', modelPerSecond: MODEL_PER_SECOND, windowSize: WINDOW,
    step, reset, onFrame: frame,
  });
  frame();
  hero.onResize(frame);

  for (const c of [cR, cTau, cTh, cReset, cRest, cIbase, cIamp, cARate, cWin, cUpd, cDecay, cTarget, cEta]) c.onInput(live);
  for (const c of [cSeed, cType]) c.onInput(rebuild);

  return { destroy() {} };
}
