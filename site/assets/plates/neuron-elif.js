import { DEFAULTS, elifStep, elifExpTerm, elifPhaseLine, elifFixedPoints, elifNarration, dropletCount } from '../js/models/neurons.js';
import { computeTicks } from '../js/core.js';

export const meta = {
  id: 'neuron-elif',
  title: 'Exponential integrate-and-fire',
  conex: ['ELIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const DT = 0.1;      // ms, fixed integration step
const WINDOW = 100;  // ms of model time kept on screen
const MODEL_PER_SECOND = 100;

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
      v_rest: cRest.get(), delta: cDelta.get(), theta_rh: cTheta.get(), dt: DT,
    };
  }

  // --- live model state (the real ELIF, stepped one dt at a time) ---
  let state = { v: params().v_rest };
  let t = 0;
  let lastSpikeT = null;
  const ring = []; // {t, v, spike} for the last WINDOW ms
  const rng = ui.rng(1); // seeded rng for the decorative droplets only

  function step() {
    const p = params();
    const I = cI.get();
    const r = elifStep(state, p, I, DT);
    state = { v: r.v };
    t += DT;
    if (r.spike) lastSpikeT = t;
    ring.push({ t, v: r.v, spike: r.spike });
    while (ring.length && ring[0].t < t - WINDOW) ring.shift();
  }

  function reset() {
    state = { v: params().v_rest };
    t = 0;
    lastSpikeT = null;
    ring.length = 0;
    drops.length = 0;
  }

  // Phase line curve f(v) = dv/dt for the current params and I. Recomputed only
  // when a parameter changes; the ball just moves along it each frame. The x
  // range is v_rest-5 .. threshold and the y range is clipped to -2..+4 mV/ms so
  // the linear leak part is visible (the exponential take-off leaves the top).
  let phase = null;
  function computePhase() {
    const p = params();
    const I = cI.get();
    const vMin = p.v_rest - 5;
    const vMax = p.threshold;
    const N = 200;
    const xs = [], ys = [];
    for (let k = 0; k <= N; k++) {
      const v = vMin + (vMax - vMin) * k / N;
      xs.push(v); ys.push(elifPhaseLine(v, p, I));
    }
    const fixed = elifFixedPoints(p, I, vMin, vMax);
    phase = { xs, ys, fixed, vMin, vMax, yLo: -2, yHi: 4 };
  }

  // --- figures ---
  ui.tryThis(figures, 'Try this: lower <var>theta</var><sub>rh</sub>; watch the take-off start earlier and the ball accelerate sooner.');
  const hero = ui.hero(figures, { height: 340 });
  const figV = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });

  // --- notes ---
  ui.note(notes, `The ELIF adds an exponential spike-initiation term to the LIF leak. As the voltage
  approaches the boosting threshold <var>theta</var><sub>rh</sub>, the term <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>)
  grows sharply, so the neuron accelerates toward threshold on its own. <var>delta</var> sets how sharp that
  take-off is; <var>theta</var><sub>rh</sub> sets where it starts. Units: v in mV, t in ms, I in pA, R in GOhm. The
  spike-detection threshold (0 mV here) is a simulation choice — Gerstner gives no value for it.
  The hero shows the phase line (the ball is the live v rolling along dv/dt) and the bucket with an
  extra "pump" stream for the exponential term.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>delta</var>·exp((<var>v</var>−<var>theta</var><sub>rh</sub>)/<var>delta</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The phase line plots d<var>v</var>/d<var>t</var> against <var>v</var> for the current input. Where the curve
  crosses zero is a fixed point: if it is below threshold the neuron settles there (no firing); if the
  input is strong enough that no stable fixed point sits below threshold, the voltage runs away and fires.
  The shaded band is the take-off region (about <var>theta</var><sub>rh</sub> to <var>theta</var><sub>rh</sub>+3<var>delta</var>), where the
  exponential term dominates. In the voltage trace the spike peaks look like they only reach about
  −25 mV even though the detection threshold is 0 mV: at dt=0.1 ms the upstroke is a single Euler
  step, so the step that crosses threshold is reset before it is recorded.
  ML analogy: the exponential term is like a soft, self-reinforcing "excitation" — the analogy breaks
  because it is a deterministic dynamical term, not a learned weight.`);
  ui.code(notes, () => codeSnippet());

  // --- hero drawing: phase line (left) + bucket with pump (right) ---
  const drops = []; // decorative droplets {x,y,vx,vy,life,maxLife,kind}
  let lastReal = 0;

  function spawnDrops(n, x, y, vx, vy, kind) {
    for (let i = 0; i < n; i++) {
      drops.push({ x, y, vx: vx + (rng.next() - 0.5) * 12, vy, life: 0, maxLife: 0.35 + rng.next() * 0.3, kind });
    }
    if (drops.length > 200) drops.splice(0, drops.length - 200);
  }

  function tickDrops(dtReal) {
    for (let i = drops.length - 1; i >= 0; i--) {
      const d = drops[i];
      d.life += dtReal;
      d.x += d.vx * dtReal;
      d.y += d.vy * dtReal;
      d.vy += 260 * dtReal;
      if (d.life > d.maxLife) drops.splice(i, 1);
    }
  }

  function drawHero() {
    const p = params();
    const I = cI.get();
    const vNow = state.v;
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
      pump: ui.color('c6'), coral: ui.color('c2'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const splashing = vNow <= p.v_reset + 1e-9;

    // ---- left: phase line ----
    const split = W * 0.56;
    const M = { left: 40, right: 8, top: 16, bottom: 30 };
    const sx = (x) => M.left + ((x - phase.vMin) / (phase.vMax - phase.vMin)) * (split - M.left - M.right);
    const sy = (y) => (H - M.bottom) - ((y - phase.yLo) / (phase.yHi - phase.yLo)) * (H - M.bottom - M.top);
    // take-off band: theta_rh .. theta_rh + 3*delta, on the SAME x scale as the curve
    const bandL = Math.max(phase.vMin, p.theta_rh);
    const bandR = Math.min(phase.vMax, p.theta_rh + 3 * p.delta);
    if (bandR > bandL) {
      ctx.fillStyle = pal.pump;
      ctx.globalAlpha = 0.12;
      ctx.fillRect(sx(bandL), M.top, sx(bandR) - sx(bandL), H - M.bottom - M.top);
      ctx.globalAlpha = 1;
    }
    // frame
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1;
    ctx.strokeRect(M.left, M.top, split - M.left - M.right, H - M.bottom - M.top);
    // x tick labels (v, mV): pick the count from the available width so labels
    // are at least 40 px apart (a fixed step of 10 collides at 390 px).
    const axisW = split - M.left - M.right;
    const xTarget = Math.max(2, Math.floor(axisW / 40) + 1);
    const xTicks = computeTicks(phase.vMin, phase.vMax, xTarget);
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const xv of xTicks.ticks) {
      ctx.fillText(String(xv), sx(xv), H - M.bottom + 3);
    }
    // y tick labels (dv/dt, mV/ms)
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let yv = Math.ceil(phase.yLo / 2) * 2; yv <= phase.yHi; yv += 2) {
      ctx.fillText(String(yv), M.left - 4, sy(yv));
    }
    // zero line
    ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(M.left, sy(0)); ctx.lineTo(split - M.right, sy(0)); ctx.stroke();
    // theta_rh marker
    if (p.theta_rh >= phase.vMin && p.theta_rh <= phase.vMax) {
      ctx.beginPath(); ctx.moveTo(sx(p.theta_rh), M.top); ctx.lineTo(sx(p.theta_rh), H - M.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('theta_rh', sx(p.theta_rh) + 2, M.top + 2);
      ctx.setLineDash([4, 3]);
    }
    ctx.setLineDash([]);
    // the curve f(v)
    ctx.strokeStyle = pal.water; ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let k = 0; k < phase.xs.length; k++) {
      const yv = phase.ys[k];
      if (yv < phase.yLo - 0.5 || yv > phase.yHi + 0.5) { started = false; continue; }
      const px = sx(phase.xs[k]), py = sy(Math.max(phase.yLo, Math.min(phase.yHi, yv)));
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // fixed points: stable = filled dot, unstable = open dot, on the zero line
    for (const fp of phase.fixed) {
      if (fp.v < phase.vMin || fp.v > phase.vMax) continue;
      const fx = sx(fp.v), fy = sy(0);
      ctx.beginPath(); ctx.arc(fx, fy, 4, 0, Math.PI * 2);
      if (fp.stable) { ctx.fillStyle = pal.water; ctx.fill(); }
      else { ctx.fillStyle = ui.color('paper'); ctx.fill(); ctx.strokeStyle = pal.water; ctx.lineWidth = 1.5; ctx.stroke(); }
    }
    // the ball at the live v, on the curve (clipped to the window)
    const dNow = elifPhaseLine(vNow, p, I);
    const bx = sx(vNow), by = sy(Math.max(phase.yLo, Math.min(phase.yHi, dNow)));
    ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2);
    ctx.fillStyle = splashing ? pal.coral : pal.ink; ctx.fill();
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 1; ctx.stroke();
    // axis titles
    ctx.fillStyle = pal.ink; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('v (mV)', (M.left + split - M.right) / 2, H - 2);
    ctx.save();
    ctx.translate(10, (M.top + H - M.bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('dv/dt (mV/ms)', 0, 0);
    ctx.restore();

    // ---- right: bucket with tap (input), hole (leak), pump (exponential) ----
    const { input, leak } = { input: (p.R * I) / p.tau, leak: (vNow - p.v_rest) / p.tau };
    const pump = elifExpTerm(vNow, p) / p.tau;
    const bw = Math.min(150, Math.max(96, (W - split) * 0.5));
    const bx0 = split + (W - split - bw) / 2;
    const cx = bx0 + bw / 2;
    const yRim = 90, yBot = 280;
    const yOf = (val) => yBot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (yBot - yRim);
    const level = Math.max(0, Math.min(1, (vNow - p.v_rest) / (p.threshold - p.v_rest)));
    const yWater = yBot - level * (yBot - yRim);
    ctx.fillStyle = splashing ? pal.coral : pal.water;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(bx0 + 2, yWater, bw - 4, yBot - yWater);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = bx0 + 2; x <= bx0 + bw - 2; x += 4) {
      const y = yWater + Math.sin((x - bx0) / 9) * 1.2;
      if (x === bx0 + 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = pal.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx0, yRim); ctx.lineTo(bx0, yBot); ctx.lineTo(bx0 + bw, yBot); ctx.lineTo(bx0 + bw, yRim);
    ctx.stroke();
    const yReset = yOf(p.v_reset);
    ctx.strokeStyle = pal.muted; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(bx0, yRim); ctx.lineTo(bx0 + bw, yRim); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx0, yReset); ctx.lineTo(bx0 + bw, yReset); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.muted; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`threshold ${p.threshold} mV`, bx0 - 6, yRim);
    ctx.fillText(`v_reset ${p.v_reset} mV`, bx0 - 6, yReset);
    ctx.fillText(`v_rest ${p.v_rest} mV`, bx0 - 6, yBot);
    // tap (input) at the top-centre
    const tapW = Math.max(2, Math.min(24, 2 + input * 4));
    ctx.strokeStyle = pal.tap; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx - 14, 40); ctx.lineTo(cx + 14, 40); ctx.stroke();
    ctx.lineWidth = tapW;
    ctx.beginPath(); ctx.moveTo(cx, 44); ctx.lineTo(cx, yWater - 2); ctx.stroke();
    ctx.fillStyle = pal.tap; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`input ${input.toFixed(2)} mV/ms`, cx, 34);
    // hole (leak) on the right wall
    const holeY = yBot - 20;
    const leakW = Math.max(1.5, Math.min(24, 1.5 + leak * 4));
    const jetLen = Math.max(8, Math.min(48, 8 + leak * 30));
    ctx.strokeStyle = pal.leak; ctx.lineWidth = leakW;
    ctx.beginPath();
    ctx.moveTo(bx0 + bw - 2, holeY);
    ctx.quadraticCurveTo(bx0 + bw + jetLen * 0.5, holeY + 4, bx0 + bw + jetLen, holeY + jetLen * 0.5);
    ctx.stroke();
    ctx.fillStyle = pal.leak; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`leak ${leak.toFixed(2)} mV/ms`, cx, yBot + 12);
    // pump (exponential term) on the left wall, feeding in
    const pumpW = Math.max(1.5, Math.min(24, 1.5 + pump * 4));
    const pumpLen = Math.max(8, Math.min(48, 8 + pump * 30));
    ctx.strokeStyle = pal.pump; ctx.lineWidth = pumpW;
    ctx.beginPath();
    ctx.moveTo(bx0 + 2, yWater + 30);
    ctx.quadraticCurveTo(bx0 - pumpLen * 0.5, yWater + 30 - 4, bx0 - pumpLen, yWater + 30 - pumpLen * 0.5);
    ctx.stroke();
    ctx.fillStyle = pal.pump; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`pump ${pump.toFixed(2)} mV/ms`, cx, yBot + 28);
    // v readout: inside the water (white on the teal fill), clamped so it stays
    // in the bucket even when the level is low.
    ctx.fillStyle = '#ffffff'; ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, bx0 + 6, Math.min(yWater + 4, yBot - 14));
    // spike rim flash
    if (splashing) {
      ctx.strokeStyle = pal.coral; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(bx0, yRim); ctx.lineTo(bx0 + bw, yRim); ctx.stroke();
    }

    // decorative droplets (input from tap, leak from hole, pump from left)
    ctx.fillStyle = pal.tap;
    for (const d of drops) if (d.kind === 'tap') { ctx.globalAlpha = Math.max(0, 1 - d.life / d.maxLife); ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = pal.leak;
    for (const d of drops) if (d.kind === 'leak') { ctx.globalAlpha = Math.max(0, 1 - d.life / d.maxLife); ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = pal.pump;
    for (const d of drops) if (d.kind === 'pump') { ctx.globalAlpha = Math.max(0, 1 - d.life / d.maxLife); ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2); ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  // --- frame ---
  function frame() {
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const dtReal = lastReal ? Math.min(0.05, (now - lastReal) / 1000) : 0;
    lastReal = now;
    const p = params();
    const I = cI.get();
    const input = (p.R * I) / p.tau;
    const leak = (state.v - p.v_rest) / p.tau;
    const pump = elifExpTerm(state.v, p) / p.tau;
    const bw = Math.min(150, Math.max(96, (hero.width - hero.width * 0.56) * 0.5));
    const bx0 = hero.width * 0.56 + (hero.width * 0.44 - bw) / 2;
    const cx = bx0 + bw / 2;
    spawnDrops(dropletCount(input, dtReal, rng), cx, 44, 0, 40, 'tap');
    spawnDrops(dropletCount(leak, dtReal, rng), bx0 + bw - 2, 280 - 20, 60, 10, 'leak');
    spawnDrops(dropletCount(pump, dtReal, rng), bx0 + 2, 280 - 20 - 30, -60, -10, 'pump');
    tickDrops(dtReal);
    drawHero();

    const xs = ring.map((s) => s.t);
    const ys = ring.map((s) => s.v);
    figV.draw([{ x: xs, y: ys, label: 'v', color: 'c1', style: 'line' }],
      { xRange: [t - WINDOW, t], hlines: [{ y: p.threshold, label: 'threshold' }] });
    hero.narration(elifNarration({ v: state.v, p, I, splashing: state.v <= p.v_reset + 1e-9 }));
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import ELIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${DT})])
pop = NeuronGroup(net=net, size=1, behavior=[
    ELIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
         v_reset=${p.v_reset}, v_rest=${p.v_rest},
         delta=${p.delta}, theta_rh=${p.theta_rh}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(1000)`;
  }

  // Recompute the phase line only when a parameter changes (the ball moves each frame).
  function onParam() {
    computePhase();
    frame();
  }

  // Compute the phase line before ui.live: the runner may call onFrame at mount.
  computePhase();
  ui.live(figures, {
    dt: DT, unit: 'ms', modelPerSecond: MODEL_PER_SECOND, windowSize: WINDOW,
    step, reset, onFrame: frame,
  });
  frame();
  hero.onResize(frame);

  for (const c of [cR, cTau, cTh, cReset, cRest, cDelta, cTheta, cI]) c.onInput(onParam);

  return { destroy() {} };
}
