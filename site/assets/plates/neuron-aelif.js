import { DEFAULTS, TABLE61, aelifStep, spikeTimes, firingStats } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-aelif',
  title: 'Adaptive exponential integrate-and-fire',
  conex: ['AELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 5000;
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
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(),
      alpha: cAlpha.get(), beta: cBeta.get(), w_tau: cWTau.get(), dt: cDt.get(),
    };
  }

  const status = document.createElement('p');
  status.style.fontStyle = 'italic';
  figures.appendChild(status);

  // Compute the whole run once; the player reveals it frame by frame.
  let runData = null;
  function computeRun() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], w = [], spikes = [];
    let state = { v: p.v_rest, omega: 0.0 };
    for (let n = 0; n < STEPS; n++) {
      const r = aelifStep(state, p, I, p.dt);
      state = { v: r.v, omega: r.omega };
      t.push(n * p.dt); v.push(r.v); w.push(r.omega); spikes.push(r.spike);
    }
    runData = { p, t, v, w, spikes };

    // Full-run summary (a property of the whole run, not a per-frame value).
    const s = firingStats(spikeTimes(spikes, p.dt));
    const name = cPattern.get();
    const reproduced = !NOT_REPRODUCED.has(name);
    const first = s.first_t == null ? '—' : s.first_t.toPrecision(4);
    status.textContent = `${name}: ${s.n} spikes in ${(STEPS * p.dt).toPrecision(4)} ms, CV=${s.cv?.toPrecision(3) ?? '—'}, ` +
      `first spike ${first} ms — ${reproduced ? 'pattern reproduced' : 'not reproduced at this dt'}`;
    return t[t.length - 1];
  }

  const figV = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figW = ui.plot(figures, { height: 160, xLabel: 't', yLabel: 'w (omega)', xUnit: 'ms', yUnit: 'pA' });
  // Membrane gauge: a vertical bar from v_rest to threshold filled to the current v,
  // plus a readout of the adaptation current w at the same time.
  const gauge = ui.canvas(figures, { height: 220 });
  // Build the run before the player: ui.player may call onFrame synchronously.
  const player = ui.player(figures, { duration: computeRun(), unit: 'ms', playSeconds: 5, loop: true, onFrame: frame });

  ui.note(notes, `The AELIF adds an adaptation current <var>w</var> (called <var>omega</var> in CoNeX) to the ELIF.
  It has two drivers: a subthreshold term <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) that builds up while the neuron is
  active, and a spike-triggered kick <var>beta</var>·<var>w</var><sub>tau</sub> on every spike. The term <var>R</var>·<var>w</var> (mV) is
  subtracted from <var>R</var>·<var>I</var>, so as <var>w</var> grows the neuron fires more slowly — this is <b>adaptation</b>.`);
  ui.equation(notes, `<var>tau</var><sub>w</sub>·d<var>w</var>/d<var>t</var> = <var>alpha</var>·(<var>v</var>−<var>v</var><sub>rest</sub>) − <var>w</var> + <var>beta</var>·<var>tau</var><sub>w</sub>·spikes`);
  ui.note(notes, `The "Firing pattern" select loads the seven rows of Gerstner et al. Table 6.1 (R=0.5 GOhm,
  v<sub>rest</sub>=−70 mV, <var>theta</var><sub>rh</sub>=−50 mV, <var>delta</var>=2 mV, threshold=0 mV, dt=0.1 ms). The spike-detection
  threshold of 0 mV is a simulation choice — the book gives no value for it. With the CoNeX equations
  at dt=0.1 ms, Tonic, Adapting, Initial burst, Transient and Delayed reproduce their named patterns;
  <b>Bursting and Irregular do not</b> (both fire nearly regularly), so they are labelled "not reproduced
  at this dt". Note that CoNeX's Euler step multiplies the whole adaptation update by dt/w_tau, so
  each spike raises <var>w</var> by <var>beta</var>·dt (0.7 pA here), not by <var>beta</var> as in Gerstner's
  AdEx. With the jump set to <var>beta</var>, Bursting does burst (groups of 4-8 spikes at ~1-2 ms
  intervals separated by ~60 ms gaps) and Irregular's inter-spike-interval CV rises from 0.07 to 1.83
  (whether the pattern is irregular rather than bursting was not checked), so the small spike jump
  explains why Bursting is not reproduced. ML analogy: <var>w</var> is a short-term
  "fatigue" state like a running average of recent activity; the analogy breaks because it is a
  continuous dynamical variable, not a discrete counter.`);
  ui.code(notes, () => codeSnippet());

  function applyPattern() {
    const row = TABLE61[cPattern.get()];
    cTau.set(row.tau); cAlpha.set(row.alpha); cWTau.set(row.w_tau);
    cBeta.set(row.beta); cReset.set(row.v_reset); cI.set(row.I);
    run();
  }

  // On a parameter change: recompute the run and continue from the current t (clamped).
  function run() {
    player.setDuration(computeRun());
  }

  // Reveal the v and w traces up to time t and draw the gauge at the real v(t), w(t).
  function frame(t) {
    if (!runData) return;
    const { p, t: tt, v, w } = runData;
    figV.draw([{ x: tt, y: v, label: 'v', color: 'c1', style: 'line' }],
      { revealX: t, hlines: [{ y: p.threshold, label: 'threshold' }] });
    figW.draw([{ x: tt, y: w, label: 'w', color: 'c4', style: 'line' }], { revealX: t });
    drawGauge(t);
  }

  // Vertical bar from v_rest to threshold filled to the real v(t), with the
  // adaptation current w(t) read out beside it. Coral flash on a spike step.
  function drawGauge(t) {
    const { p, v, w } = runData;
    const idx = Math.min(v.length - 1, Math.max(0, Math.round(t / p.dt)));
    const vNow = v[idx];
    const wNow = w[idx];
    const spiked = vNow <= p.v_reset + 1e-9; // reset value means this step fired
    const ctx = gauge.ctx;
    const W = gauge.width, H = gauge.height;
    ctx.clearRect(0, 0, W, H);
    const pal = { ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule') };
    const x = W / 2, bw = 26, top = 14, bot = H - 14;
    const yOf = (val) => bot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (bot - top);
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1;
    ctx.strokeRect(x - bw / 2, top, bw, bot - top);
    const vClamped = Math.max(p.v_rest, Math.min(p.threshold, vNow));
    const yTop = yOf(vClamped);
    ctx.fillStyle = spiked ? ui.color('c2') : ui.color('c1');
    ctx.fillRect(x - bw / 2 + 1, yTop, bw - 2, bot - yTop);
    ctx.strokeStyle = pal.muted; ctx.setLineDash([3, 3]);
    for (const [val, lab] of [[p.threshold, 'threshold'], [p.v_rest, 'v_rest']]) {
      const y = yOf(val);
      ctx.beginPath(); ctx.moveTo(x - bw / 2 - 4, y); ctx.lineTo(x + bw / 2 + 4, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted;
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(`${lab} ${val} mV`, x + bw / 2 + 8, y);
      ctx.setLineDash([3, 3]);
    }
    ctx.setLineDash([]);
    // v readout above the bar (the w readout is top-left, the markers are on the right).
    ctx.fillStyle = pal.ink;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, x, top - 4);
    ctx.fillStyle = ui.color('c4');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`w = ${wNow.toFixed(2)} pA`, 6, 6);
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import AELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    AELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
          v_reset=${p.v_reset}, v_rest=${p.v_rest},
          delta=${p.delta}, theta_rh=${p.theta_rh},
          alpha=${p.alpha}, beta=${p.beta}, w_tau=${p.w_tau}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  cPattern.onInput(applyPattern);
  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cAlpha, cBeta, cWTau, cDt, cI]) c.onInput(run);
  ui.onTheme(run);

  return { destroy() {} };
}
