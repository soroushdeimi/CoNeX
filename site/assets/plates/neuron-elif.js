import { DEFAULTS, elifStep } from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-elif',
  title: 'Exponential integrate-and-fire',
  conex: ['ELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 500;

// dv/dt for the ELIF at a given v and constant current I (mV per ms).
function dvdt(v, p, I) {
  const Fu = p.v_rest - v + p.delta * Math.exp((v - p.theta_rh) / p.delta);
  return (Fu + p.R * I) / p.tau;
}

// Use ui.player when the shell provides it; otherwise fall back to a stub that
// reveals the whole run once, so figures are never empty in the test mock.
function makePlayer(ui, parent, onFrame) {
  if (typeof ui.player === 'function') {
    return ui.player(parent, { duration: 0, unit: 'ms', playSeconds: 5, loop: true, onFrame });
  }
  let t = 0;
  return {
    play() {}, pause() {},
    seek(x) { t = x; }, // no player to animate; setDuration reveals the full run
    playing() { return false; },
    setDuration(d) { t = d; onFrame(d); },
  };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -20, max: 20, step: 0.5, value: P.threshold });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDelta = ui.slider(sec, { key: 'delta', label: 'delta', unit: 'mV', min: 0.5, max: 10, step: 0.1, value: P.delta });
  const cTheta = ui.slider(sec, { key: 'theta_rh', label: 'theta_rh', unit: 'mV', min: -70, max: -30, step: 0.5, value: P.theta_rh });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });
  const cI = ui.slider(sec, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  // Phase line on a canvas so the moving dot and its dv/dt arrow can be overlaid.
  const phase = ui.canvas(figures, { height: 240 });
  const phaseCap = document.createElement('p');
  phaseCap.style.fontStyle = 'italic';
  phaseCap.textContent = 'Phase line: dv/dt (mV/ms) vs v (mV); the dot is the current v(t) and the arrow is dv/dt.';
  figures.appendChild(phaseCap);
  const figV = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  // Membrane gauge: a vertical bar from v_rest to threshold filled to the current v.
  const gauge = ui.canvas(figures, { height: 240 });
  const player = makePlayer(ui, figures, frame);

  ui.note(notes, `The ELIF adds an exponential spike-initiation term to the LIF leak. As the voltage
  approaches the boosting threshold <var>theta</var><sub>rh</sub>, the term <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>)
  grows sharply, so the neuron accelerates toward threshold on its own. <var>delta</var> sets how sharp that
  take-off is; <var>theta</var><sub>rh</sub> sets where it starts. Units: v in mV, t in ms, I in pA, R in GOhm. The
  spike-detection threshold (0 mV here) is a simulation choice — Gerstner gives no value for it.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The phase line plots d<var>v</var>/d<var>t</var> against <var>v</var> for the current input. Where the curve
  crosses zero is a fixed point: if it is below threshold the neuron settles there (no firing); if the
  input is strong enough that no stable fixed point sits below threshold, the voltage runs away and fires.
  The vertical axis is clipped to a readable window; above <var>theta</var><sub>rh</sub> the exponential term makes
  d<var>v</var>/d<var>t</var> grow without bound, so the curve leaves the top of the window there.
  In the voltage trace the spike peaks look like they only reach about −25 mV even though the
  detection threshold is 0 mV: at dt=0.1 ms the upstroke is a single Euler step, so the step that
  crosses threshold is reset before it is recorded and the plotted peak is the last value below
  threshold, not the true peak.
  ML analogy: the exponential term is like a soft, self-reinforcing "excitation" — the analogy breaks
  because it is a deterministic dynamical term, not a learned weight.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(), dt: cDt.get(),
    };
  }

  let runData = null;

  // Recompute the whole run once; the player then reveals it frame by frame.
  function run() {
    const p = params();
    const I = cI.get();

    // Phase line: dv/dt over a range of v, mark zero crossings (fixed points).
    const vMin = Math.min(p.v_rest, p.v_reset) - 5;
    const vMax = p.threshold + 5;
    const N = 200;
    const xs = [], ys = [];
    let prev = null;
    const fixed = [];
    for (let k = 0; k <= N; k++) {
      const v = vMin + (vMax - vMin) * k / N;
      const d = dvdt(v, p, I);
      xs.push(v); ys.push(d);
      if (prev !== null && (prev.d <= 0) !== (d <= 0)) {
        const t = prev.d / (prev.d - d);
        fixed.push(prev.v + t * (v - prev.v));
      }
      prev = { v, d };
    }
    // Clip the y window to the interesting part: the exponential term blows up
    // above ~theta_rh, so a fixed window keeps the fixed points readable.
    const yLo = -2, yHi = 10; // mV/ms

    // Voltage trace.
    const t = [], v = [];
    let state = { v: p.v_rest };
    for (let n = 0; n < STEPS; n++) {
      const r = elifStep(state, p, I, p.dt);
      state = { v: r.v };
      t.push(n * p.dt); v.push(r.v);
    }
    runData = { p, I, xs, ys, fixed, vMin, vMax, yLo, yHi, t, v };
    player.setDuration(t[t.length - 1]);
    player.seek(0); // reveals the first frame (also covers reduced-motion, which starts paused)
  }

  // Reveal the voltage trace up to time t, draw the gauge, and move the phase dot.
  function frame(t) {
    if (!runData) return;
    const { p, t: tt, v } = runData;
    figV.draw([{ x: tt, y: v, label: 'v', color: 'c2', style: 'line' }],
      { revealX: t, hlines: [{ y: p.threshold, label: 'threshold' }] });
    drawGauge(t);
    drawPhase(t);
  }

  // Phase line drawn on a canvas so the moving dot and its dv/dt arrow can be
  // overlaid on the curve. The curve itself is the real dv/dt(v) for the current I.
  function drawPhase(t) {
    const { p, I, xs, ys, fixed, vMin, vMax, yLo, yHi, v } = runData;
    const idx = Math.min(v.length - 1, Math.max(0, Math.round(t / p.dt)));
    const vNow = v[idx];
    const dNow = dvdt(vNow, p, I);
    const ctx = phase.ctx;
    const W = phase.width, H = phase.height;
    const M = { left: 44, right: 12, top: 12, bottom: 30 };
    ctx.clearRect(0, 0, W, H);
    const pal = { ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule') };
    const sx = (x) => M.left + ((x - vMin) / (vMax - vMin)) * (W - M.left - M.right);
    const sy = (y) => (H - M.bottom) - ((y - yLo) / (yHi - yLo)) * (H - M.bottom - M.top);
    // frame + ticks
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = pal.muted;
    for (let gx = Math.ceil(vMin / 10) * 10; gx <= vMax; gx += 10) {
      ctx.beginPath(); ctx.moveTo(sx(gx), M.top); ctx.lineTo(sx(gx), H - M.bottom); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(gx, sx(gx), H - M.bottom + 3);
    }
    for (let gy = Math.ceil(yLo / 2) * 2; gy <= yHi; gy += 2) {
      ctx.beginPath(); ctx.moveTo(M.left, sy(gy)); ctx.lineTo(W - M.right, sy(gy)); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(gy, M.left - 4, sy(gy));
    }
    ctx.strokeStyle = pal.ink; ctx.strokeRect(M.left, M.top, W - M.left - M.right, H - M.bottom - M.top);
    // zero line
    ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(M.left, sy(0)); ctx.lineTo(W - M.right, sy(0)); ctx.stroke();
    // theta_rh and fixed points
    for (const [xv, lab] of [[p.theta_rh, 'theta_rh'], ...fixed.map((f) => [f, 'fixed'])]) {
      ctx.beginPath(); ctx.moveTo(sx(xv), M.top); ctx.lineTo(sx(xv), H - M.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(lab, sx(xv) + 2, M.top + 2);
      ctx.setLineDash([4, 3]);
    }
    ctx.setLineDash([]);
    // the curve
    ctx.strokeStyle = ui.color('c1'); ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k < xs.length; k++) {
      const px = sx(xs[k]), py = sy(ys[k]);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // moving dot at the current v, with an arrow showing dv/dt (vertical, up if rising)
    const dx = sx(vNow), dy = sy(Math.max(yLo, Math.min(yHi, dNow)));
    const arrowLen = Math.max(-40, Math.min(40, dNow * 4)); // px, clipped to the window
    ctx.strokeStyle = ui.color('c2'); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(dx, dy - arrowLen); ctx.stroke();
    const ah = 5;
    ctx.beginPath();
    ctx.moveTo(dx, dy - arrowLen);
    ctx.lineTo(dx - ah, dy - arrowLen + (arrowLen >= 0 ? ah : -ah));
    ctx.lineTo(dx + ah, dy - arrowLen + (arrowLen >= 0 ? ah : -ah));
    ctx.closePath(); ctx.fillStyle = ui.color('c2'); ctx.fill();
    ctx.beginPath(); ctx.arc(dx, dy, 3.5, 0, Math.PI * 2); ctx.fillStyle = ui.color('c2'); ctx.fill();
    // axis titles
    ctx.fillStyle = pal.ink; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('v (mV)', (M.left + W - M.right) / 2, H - 2);
  }

  // Vertical bar from v_rest to threshold; the fill is the real v at time t.
  // Coral flash on the step where the neuron just spiked.
  function drawGauge(t) {
    const { p, v } = runData;
    const idx = Math.min(v.length - 1, Math.max(0, Math.round(t / p.dt)));
    const vNow = v[idx];
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
    ctx.fillStyle = pal.ink;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, x, bot + 4);
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import ELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    ELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
         v_reset=${p.v_reset}, v_rest=${p.v_rest},
         delta=${p.delta}, theta_rh=${p.theta_rh}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
