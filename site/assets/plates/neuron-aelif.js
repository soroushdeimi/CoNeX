import { DEFAULTS, TABLE61, aelifStep, spikeTimes, firingStats, aelifNarration, dropletCount } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-aelif',
  title: 'Adaptive exponential integrate-and-fire',
  conex: ['AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const DT = 0.1;      // ms, fixed integration step
const WINDOW = 100;  // ms of model time kept on screen
const MODEL_PER_SECOND = 100;
// Rows whose named pattern is NOT reproduced by CoNeX's AELIF at dt=0.1 ms.
const NOT_REPRODUCED = new Set(['Bursting', 'Irregular']);

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const secP = ui.section(controls, 'Firing pattern (Gerstner Table 6.1)');
  const cPattern = ui.select(secP, {
    key: 'pattern', label: 'pattern',
    options: Object.keys(TABLE61).map((k) => ({ value: k, label: k })),
    value: 'Tonic',
  });

  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -20, max: 20, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -70, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'mV', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'mV', min: -70, max: -30, step: 0.5, value: P.theta_rh });
  const cAlpha = ui.slider(sec, { key: 'alpha', label: 'alpha', unit: 'nS', min: -2, max: 2, step: 0.05, value: P.alpha });
  const cBeta = ui.slider(sec, { key: 'beta', label: 'beta', unit: 'pA', min: 0, max: 100, step: 1, value: P.beta });
  const cWTau = ui.slider(sec, { key: 'w_tau', label: 'w_tau', unit: 'ms', min: 5, max: 300, step: 5, value: P.w_tau });
  const dtNote = document.createElement('p');
  dtNote.style.fontStyle = 'italic';
  dtNote.style.color = 'var(--muted)';
  dtNote.style.fontSize = '0.85rem';
  dtNote.textContent = `dt = ${DT} ms (fixed integration step; the other parameters update live)`;
  sec.appendChild(dtNote);
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: DT,
    };
  }

  const status = document.createElement('p');
  status.style.fontStyle = 'italic';
  figures.appendChild(status);

  // --- live model state (the real AELIF, stepped one dt at a time) ---
  let state = { v: params().v_rest, omega: 0.0 };
  let t = 0;
  let lastSpikeT = null;
  const ring = []; // {t, v, omega, spike} for the last WINDOW ms

  function step() {
    const p = params();
    const I = cI.get();
    const r = aelifStep(state, p, I, DT);
    state = { v: r.v, omega: r.omega };
    t += DT;
    if (r.spike) lastSpikeT = t;
    ring.push({ t, v: r.v, omega: r.omega, spike: r.spike });
    while (ring.length && ring[0].t < t - WINDOW) ring.shift();
  }

  function reset() {
    state = { v: params().v_rest, omega: 0.0 };
    t = 0;
    lastSpikeT = null;
    ring.length = 0;
  }

  // --- figures ---
  const tryThis = document.createElement('p');
tryThis.style.fontStyle = 'italic';
  tryThis.style.color = 'var(--muted)';
  tryThis.style.fontSize = '0.9rem';
  tryThis.innerHTML = '<b>Try this:</b> raise <var>beta</var>; watch the w tank jump higher on each spike and the firing slow down.';
  figures.appendChild(tryThis);
  const hero = ui.hero(figures, { height: 340 });
  const figV = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figW = ui.plot(figures, { height: 140, xLabel: 't', yLabel: 'w (omega)', xUnit: 'ms', yUnit: 'pA' });

  // --- notes ---
  ui.note(notes, `The AELIF adds an adaptation current <var>w</var> (called <var>omega</var> in CoNeX) to the ELIF.
  It has two drivers: a subthreshold term <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) that builds up while the neuron is
  active, and a spike-triggered kick on every spike. The term <var>R</var>·<var>w</var> (mV) is
  subtracted from <var>R</var>·<var>I</var>, so as <var>w</var> grows the neuron fires more slowly — this is <b>adaptation</b>.
  The hero shows the membrane bucket (v) and a second small tank (w) that rises on each spike and
  drains with <var>w</var><sub>tau</sub>; a pipe from the w tank drains the main bucket with flow <var>R</var>·<var>w</var>.`);
  ui.equation(notes, `<var>tau</var><sub>w</sub>·d<var>w</var>/d<var>t</var> = <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) − <var>w</var> + <var>beta</var>·<var>tau</var><sub>w</sub>·spikes`);
  ui.note(notes, `The "Firing pattern" select loads the seven rows of Gerstner et al. Table 6.1 (R=0.5 GOhm,
  v<sub>rest</sub>=−70 mV, <var>theta</var><sub>rh</sub>=−50 mV, <var>delta</var>=2 mV, threshold=0 mV, dt=0.1 ms). The spike-detection
  threshold of 0 mV is a simulation choice — the book gives no value for it. With the CoNeX equations
  at dt=0.1 ms, Tonic, Adapting, Initial burst, Transient and Delayed reproduce their named patterns;
  <b>Bursting and Irregular do not</b> (both fire nearly regularly), so they are labelled "not reproduced
  at this dt". Note that CoNeX's Euler step multiplies the whole adaptation update by dt/w_tau, so
  each spike raises <var>w</var> by <var>beta</var>·dt (0.7 pA here), not by <var>beta</var> as in Gerstner's
  AdEx. ML analogy: <var>w</var> is a short-term "fatigue" state like a running average of recent
  activity; the analogy breaks because it is a continuous dynamical variable, not a discrete counter.`);
  ui.code(notes, () => codeSnippet());

  function applyPattern() {
    const row = TABLE61[cPattern.get()];
    cTau.set(row.tau); cAlpha.set(row.alpha); cWTau.set(row.w_tau);
    cBeta.set(row.beta); cReset.set(row.v_reset); cI.set(row.I);
    run();
  }

  // --- hero drawing: membrane bucket (v) + adaptation tank (w) + pipe ---
  function drawHero() {
    const p = params();
    const I = cI.get();
    const vNow = state.v;
    const wNow = state.omega;
    const ctx = hero.ctx;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    if (p.threshold <= p.v_rest) {
      ctx.fillStyle = ui.color('muted');
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('threshold must exceed v_rest for the picture', W / 2, H / 2);
      return;
    }
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), tap: ui.color('c3'), leak: ui.color('c4'),
      adapt: ui.color('c4'), coral: ui.color('c2'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const splashing = vNow <= p.v_reset + 1e-9;

    // ---- main bucket (v) on the left ----
    // Reserve a left margin wide enough for the level labels (measured), so they
    // are never clipped at 390 px. Fall back to a fixed width if measureText is
    // unavailable (e.g. in the test mock).
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const tw_ = (s) => { const m = ctx.measureText(s); return (m && typeof m.width === 'number') ? m.width : 0; };
    const labelW = Math.max(
      tw_(`threshold ${p.threshold} mV`),
      tw_(`v_reset ${p.v_reset} mV`),
      tw_(`v_rest ${p.v_rest} mV`),
      88, // fallback so the margin is never zero
    );
    const bx = labelW + 12;
    const bw = Math.min(180, Math.max(110, (W * 0.62 - bx) * 0.55));
    const cx = bx + bw / 2;
    const yRim = 90, yBot = 280;
    const yOf = (val) => yBot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (yBot - yRim);
    const level = Math.max(0, Math.min(1, (vNow - p.v_rest) / (p.threshold - p.v_rest)));
    const yWater = yBot - level * (yBot - yRim);
    ctx.fillStyle = splashing ? pal.coral : pal.water;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(bx + 2, yWater, bw - 4, yBot - yWater);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = bx + 2; x <= bx + bw - 2; x += 4) {
      const y = yWater + Math.sin((x - bx) / 9) * 1.2;
      if (x === bx + 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, yRim); ctx.lineTo(bx, yBot); ctx.lineTo(bx + bw, yBot); ctx.lineTo(bx + bw, yRim);
    ctx.stroke();
    const yReset = yOf(p.v_reset);
    ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(bx, yRim); ctx.lineTo(bx + bw, yRim); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx, yReset); ctx.lineTo(bx + bw, yReset); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`threshold ${p.threshold} mV`, bx - 6, yRim);
    ctx.fillText(`v_reset ${p.v_reset} mV`, bx - 6, yReset);
    ctx.fillText(`v_rest ${p.v_rest} mV`, bx - 6, yBot);
    // tap (input)
    const input = (p.R * I) / p.tau;
    const tapW = Math.max(2, Math.min(24, 2 + input * 4));
    ctx.strokeStyle = pal.tap; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx - 14, 40); ctx.lineTo(cx + 14, 40); ctx.stroke();
    ctx.lineWidth = tapW;
    ctx.beginPath(); ctx.moveTo(cx, 44); ctx.lineTo(cx, yWater - 2); ctx.stroke();
    ctx.fillStyle = pal.tap; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`input R·I/τ = ${input.toFixed(2)} mV/ms`, cx, 34);
    // v readout inside the water (white on the teal fill), clamped so it stays
    // in the bucket even when the level is low.
    ctx.fillStyle = '#ffffff'; ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, bx + 6, Math.min(yWater + 4, yBot - 14));
    // spike rim flash
    if (splashing) {
      ctx.strokeStyle = pal.coral; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(bx, yRim); ctx.lineTo(bx + bw, yRim); ctx.stroke();
    }

    // ---- adaptation tank (w) on the right ----
    // w can be negative (e.g. Delayed); map a symmetric range around 0.
    const wMax = Math.max(10, Math.abs(wNow) * 1.2);
    const tw = Math.min(120, Math.max(72, W * 0.20));
    const tx = Math.min(W - tw - 8, bx + bw + 24);
    const tcx = tx + tw / 2;
    const tTop = 110, tBot = 280;
    const tMid = (tTop + tBot) / 2;
    const wLevel = Math.max(-1, Math.min(1, wNow / wMax));
    const yW = tMid - wLevel * (tBot - tMid) * 0.9;
    // tank walls
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tx, tTop); ctx.lineTo(tx, tBot); ctx.lineTo(tx + tw, tBot); ctx.lineTo(tx + tw, tTop);
    ctx.stroke();
    // fill from the midline (w=0) up (positive) or down (negative)
    ctx.fillStyle = pal.adapt;
    ctx.globalAlpha = 0.7;
    if (wLevel >= 0) ctx.fillRect(tx + 2, yW, tw - 4, tMid - yW);
    else ctx.fillRect(tx + 2, tMid, tw - 4, yW - tMid);
    ctx.globalAlpha = 1;
    // midline (w = 0)
    ctx.strokeStyle = pal.muted; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(tx, tMid); ctx.lineTo(tx + tw, tMid); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('w = 0 pA', tx + tw + 4, tMid);
    ctx.fillStyle = pal.adapt; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`w = ${wNow.toFixed(2)} pA`, tcx, tTop - 6);
    ctx.fillStyle = pal.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('adaptation tank (w)', tcx, tBot + 6);

    // ---- pipe from the w tank draining the main bucket, flow R*w ----
    // The adaptation current subtracts R*w from the input; draw a pipe from the
    // w tank to the main bucket whose width scales with R*|w|.
    const adaptFlow = p.R * Math.abs(wNow); // mV
    const pipeW = Math.max(1.5, Math.min(20, 1.5 + adaptFlow * 0.4));
    const pipeY = tMid;
    ctx.strokeStyle = pal.adapt; ctx.lineWidth = pipeW;
    ctx.beginPath();
    ctx.moveTo(tx, pipeY);
    ctx.quadraticCurveTo((tx + bx + bw) / 2, pipeY - 30, bx + bw, yWater + 20);
    ctx.stroke();
    // The R·w value is a second caption line under the tank (the gap between the
    // bucket and the tank is too narrow to hold it between them).
    ctx.fillStyle = pal.adapt; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`drains R·w = ${(p.R * wNow).toFixed(2)} mV`, tcx, tBot + 18);
  }

  // --- frame ---
  function frame() {
    const p = params();
    const I = cI.get();
    drawHero();
    const xs = ring.map((s) => s.t);
    const ys = ring.map((s) => s.v);
    const ws = ring.map((s) => s.omega);
    figV.draw([{ x: xs, y: ys, label: 'v', color: 'c1', style: 'line' }],
      { xRange: [t - WINDOW, t], hlines: [{ y: p.threshold, label: 'threshold' }] });
    figW.draw([{ x: xs, y: ws, label: 'w', color: 'c4', style: 'line' }], { xRange: [t - WINDOW, t] });
    hero.narration(aelifNarration({ v: state.v, omega: state.omega, p, I, splashing: state.v <= p.v_reset + 1e-9 }));
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${DT})])
pop = NeuronGroup(net=net, size=1, behavior=[
    AELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
          v_reset=${p.v_reset}, v_rest=${p.v_rest},
          delta=${p.delta}, theta_rh=${p.theta_rh},
          alpha=${p.alpha}, beta=${p.beta}, w_tau=${p.w_tau}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(1000)`;
  }

  // On a parameter change: the live step picks up the new params on the next
  // step (no reset). The pattern select also sets the sliders, then run().
  function run() {
    // Update the full-run status line from a fresh precomputed run.
    const p = params();
    const I = cI.get();
    const r = simulateAelif(p, I, 5000);
    const s = firingStats(spikeTimes(r.spikes, DT));
    const name = cPattern.get();
    const reproduced = !NOT_REPRODUCED.has(name);
    const first = s.first_t == null ? '—' : s.first_t.toPrecision(4);
    status.textContent = `${name}: ${s.n} spikes in ${(5000 * DT).toPrecision(4)} ms, CV=${s.cv?.toPrecision(3) ?? '—'}, ` +
      `first spike ${first} ms — ${reproduced ? 'pattern reproduced' : 'not reproduced at this dt'}`;
    frame();
  }

  function simulateAelif(p, I, steps) {
    let st = { v: p.v_rest, omega: 0.0 };
    const spikes = [];
    for (let i = 0; i < steps; i++) {
      const r = aelifStep(st, p, I, DT);
      st = { v: r.v, omega: r.omega };
      spikes.push(r.spike);
    }
    return { spikes };
  }

  ui.live(figures, {
    dt: DT, unit: 'ms', modelPerSecond: MODEL_PER_SECOND, windowSize: WINDOW,
    step, reset, onFrame: frame,
  });
  run();
  hero.onResize(frame);

  cPattern.onInput(applyPattern);
  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cI]) c.onInput(run);

  return { destroy() {} };
}
