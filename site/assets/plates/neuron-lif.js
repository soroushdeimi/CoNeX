import {
  DEFAULTS,
  lifStep,
  lifRheobase,
  bucketFlows,
  bucketLevel,
  splashActive,
  lifNarration,
  dropletCount,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-lif',
  title: 'Leaky integrate-and-fire',
  conex: ['LIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const DT = 0.1;      // ms, fixed integration step (the CoNeX default used by the golden tests)
const WINDOW = 100;  // ms of model time kept on screen (the rolling window)
const MODEL_PER_SECOND = 100; // at 1x speed, 100 ms of model time per real second

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- controls ---
  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const dtNote = document.createElement('p');
  dtNote.style.fontStyle = 'italic';
  dtNote.style.color = 'var(--muted)';
  dtNote.style.fontSize = '0.85rem';
  dtNote.textContent = `dt = ${DT} ms (fixed integration step; the other parameters update live)`;
  sec.appendChild(dtNote);

  const secI = ui.section(controls, 'Input current');
  const cI = ui.slider(secI, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(),
      v_reset: cReset.get(), v_rest: cRest.get(), dt: DT,
    };
  }

  // --- live model state (the real LIF, stepped one dt at a time) ---
  let state = { v: params().v_rest };
  let t = 0;
  let lastSpikeT = null;
  const ring = []; // {t, v, spike} for the last WINDOW ms
  const rng = ui.rng(1); // seeded rng for the decorative droplets only

  function step() {
    const p = params();
    const I = cI.get();
    const r = lifStep(state, p, I, DT);
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

  // --- figures ---
  // Hero: the leaky bucket (the mechanism). Plots scroll below it, same time.
  const tryThis = document.createElement('p');
tryThis.style.fontStyle = 'italic';
  tryThis.style.color = 'var(--muted)';
  tryThis.style.fontSize = '0.9rem';
  tryThis.innerHTML = '<b>Try this:</b> raise <var>I</var>; watch the bucket fill faster and spike more often.';
  figures.appendChild(tryThis);
  const hero = ui.hero(figures, { height: 344 });
  const fig = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figI = ui.plot(figures, { height: 120, xLabel: 't', yLabel: 'I', xUnit: 'ms', yUnit: 'pA' });

  // --- notes ---
  ui.note(notes, `The LIF neuron integrates its input current and leaks back to rest. CoNeX updates the
  voltage once per iteration with the Euler step below, then <code>Fire</code> resets any neuron that
  reached the threshold. CoNeX has no refractory period: the reset is instantaneous and the neuron can
  fire again on the very next step. Units: v in mV, t in ms, I in pA, R in GOhm (so R·I is in mV).
  The bucket is the live mechanism: the water level is the real v, the tap flow is R·I/τ, the hole
  flow is (v − v_rest)/τ, and a spike is the rim flash plus a drop of the level to v_reset.`);
  ui.equation(notes, `<var>tau</var>·d<var>v</var>/d<var>t</var> = (<var>v</var><sub>rest</sub> − <var>v</var>) + <var>R</var>·<var>I</var>`);
  ui.note(notes, `The <b>rheobase</b> is the smallest constant current that can make the neuron fire. For the
  LIF it is <var>I</var><sub>rheo</sub> = (<var>threshold</var> − <var>v</var><sub>rest</sub>) / <var>R</var>, because that is the current whose
  steady-state voltage <var>v</var><sub>ss</sub> = <var>v</var><sub>rest</sub> + <var>R</var>·<var>I</var> just reaches the threshold. Below it the
  voltage asymptotes to a value under threshold and never fires. The dashed line marks that steady
  state on the voltage axis.`);
  ui.note(notes, `ML analogy: the LIF is the simplest "spike" unit — like a thresholded accumulator. The
  analogy breaks because the leak (<var>tau</var>) makes it a dynamical system with memory, not a static
  threshold, and the firing rate (not the voltage) is what carries information.`);
  ui.code(notes, () => codeSnippet());

  // --- hero drawing (the leaky bucket) ---
  const drops = []; // decorative droplets {x,y,vx,vy,life,maxLife,kind}
  let ripplePhase = 0;
  let lastReal = 0;

  function spawnDrops(n, x, y, vx, vy, kind) {
    for (let i = 0; i < n; i++) {
      drops.push({ x, y, vx: vx + (rng.next() - 0.5) * 12, vy, life: 0, maxLife: 0.35 + rng.next() * 0.3, kind });
    }
    if (drops.length > 200) drops.splice(0, drops.length - 200);
  }

  function drawHero() {
    const p = params();
    const I = cI.get();
    const vNow = state.v;
    const ctx = hero.ctx;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    // The bucket geometry needs threshold > v_rest (bucketLevel's real constraint).
    // A slider can set them equal/inverted; draw a note instead of a broken bucket.
    if (p.threshold <= p.v_rest) {
      ctx.fillStyle = ui.color('muted');
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('threshold must exceed v_rest for the bucket picture', W / 2, H / 2);
      return;
    }
    const { input, leak } = bucketFlows({ R: p.R, I, tau: p.tau, v_rest: p.v_rest }, vNow);
    const splashing = splashActive(t, lastSpikeT, 0.5);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), tap: ui.color('c3'), leak: ui.color('c4'), coral: ui.color('c2'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';

    const bw = Math.min(240, Math.max(128, W * 0.36));
    const bx = (W - bw) / 2;
    const cx = bx + bw / 2;
    const yRim = 90, yBot = 280;
    const yOf = (val) => yBot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (yBot - yRim);

    // Water fill from v_rest up to the REAL current level.
    const level = bucketLevel(vNow, p);
    const yWater = yBot - level * (yBot - yRim);
    ctx.fillStyle = splashing ? pal.coral : pal.water;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(bx + 2, yWater, bw - 4, yBot - yWater);
    ctx.globalAlpha = 1;
    // animated surface ripple
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = bx + 2; x <= bx + bw - 2; x += 4) {
      const y = yWater + Math.sin((x - bx) / 9 + ripplePhase) * 1.2;
      if (x === bx + 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glass bucket: open top, faint walls.
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, yRim); ctx.lineTo(bx, yBot); ctx.lineTo(bx + bw, yBot); ctx.lineTo(bx + bw, yRim);
    ctx.stroke();

    // Dashed reference lines: threshold (rim) and v_reset.
    const yReset = yOf(p.v_reset);
    ctx.strokeStyle = pal.muted;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(bx, yRim); ctx.lineTo(bx + bw, yRim); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx, yReset); ctx.lineTo(bx + bw, yReset); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = pal.muted;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`threshold ${p.threshold} mV`, bx - 10, yRim);
    ctx.fillText(`v_reset ${p.v_reset} mV`, bx - 10, yReset);
    ctx.fillText(`v_rest ${p.v_rest} mV`, bx - 10, yBot);

    // Tap above the bucket; stream width = input flow.
    const tapW = Math.max(2, Math.min(30, 2 + input * 5));
    ctx.strokeStyle = pal.tap;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(cx - 18, 40); ctx.lineTo(cx + 18, 40); ctx.stroke();
    ctx.lineWidth = tapW;
    ctx.beginPath(); ctx.moveTo(cx, 44); ctx.lineTo(cx, yWater - 2); ctx.stroke();
    ctx.fillStyle = pal.tap;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`input R·I/τ = ${input.toFixed(2)} mV/ms`, cx, 34);

    // Hole near the bottom of the right wall; jet width/length = leak flow.
    const holeY = yBot - 20;
    const leakW = Math.max(1.5, Math.min(30, 1.5 + leak * 5));
    const jetLen = Math.max(10, Math.min(70, 10 + leak * 40));
    ctx.strokeStyle = pal.leak;
    ctx.lineWidth = leakW;
    ctx.beginPath();
    ctx.moveTo(bx + bw - 2, holeY);
    ctx.quadraticCurveTo(bx + bw + jetLen * 0.5, holeY + 4, bx + bw + jetLen, holeY + jetLen * 0.5);
    ctx.stroke();
    ctx.fillStyle = pal.leak;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`leak (v − v_rest)/τ = ${leak.toFixed(2)} mV/ms`, cx, yBot + 14);

    // v readout on the water surface.
    ctx.fillStyle = pal.ink;
    ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, bx + 8, yWater - 4);

    // Decorative droplets: falling from the tap (input) and leaving the hole (leak).
    ctx.fillStyle = pal.tap;
    for (const d of drops) {
      if (d.kind === 'leak') continue;
      ctx.globalAlpha = Math.max(0, 1 - d.life / d.maxLife);
      ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = pal.leak;
    for (const d of drops) {
      if (d.kind !== 'leak') continue;
      ctx.globalAlpha = Math.max(0, 1 - d.life / d.maxLife);
      ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Spike: rim flash + spray over the rim while splashing.
    if (splashing) {
      ctx.strokeStyle = pal.coral;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(bx, yRim); ctx.lineTo(bx + bw, yRim); ctx.stroke();
      ctx.fillStyle = pal.coral;
      for (const [dx, dy] of [[-24, -14], [-8, -22], [8, -20], [24, -12], [34, -6]]) {
        ctx.beginPath(); ctx.arc(cx + dx, yRim + dy, 3, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Spike strip: a tick at each real spike in the window.
    const sx0 = bx, sx1 = bx + bw, sy = 328, sh = 16;
    ctx.strokeStyle = pal.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(sx0, sy, sx1 - sx0, sh);
    ctx.fillStyle = pal.ink;
    for (const s of ring) {
      if (!s.spike) continue;
      const px = sx0 + ((s.t - (t - WINDOW)) / WINDOW) * (sx1 - sx0);
      ctx.fillRect(px - 0.75, sy + 2, 1.5, sh - 4);
    }
    ctx.fillStyle = pal.muted;
    ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('spikes (last 100 ms)', sx0, sy - 3);
  }

  // Advance the decorative droplets and ripple by the real frame time.
  function tickDrops(dtReal) {
    ripplePhase += dtReal * 6;
    for (let i = drops.length - 1; i >= 0; i--) {
      const d = drops[i];
      d.life += dtReal;
      d.x += d.vx * dtReal;
      d.y += d.vy * dtReal;
      d.vy += 260 * dtReal;
      if (d.life > d.maxLife) drops.splice(i, 1);
    }
  }

  // --- frame: draw the hero, the rolling plots, and the narration ---
  function frame() {
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const dtReal = lastReal ? Math.min(0.05, (now - lastReal) / 1000) : 0;
    lastReal = now;

    const p = params();
    const I = cI.get();
    const { input, leak } = bucketFlows({ R: p.R, I, tau: p.tau, v_rest: p.v_rest }, state.v);
    const bw = Math.min(240, Math.max(128, hero.width * 0.36));
    const bx = (hero.width - bw) / 2;
    // spawn decorative droplets at a rate proportional to the real flows
    spawnDrops(dropletCount(input, dtReal, rng), hero.width / 2, 44, 0, 40, 'tap');
    spawnDrops(dropletCount(leak, dtReal, rng), bx + bw - 2, 280 - 20, 60, 10, 'leak');
    tickDrops(dtReal);
    drawHero();

    const xs = ring.map((s) => s.t);
    const ys = ring.map((s) => s.v);
    const ir = lifRheobase(p);
    const vss = p.v_rest + p.R * I;
    fig.draw([{ x: xs, y: ys, label: 'v', color: 'c1', style: 'line' }],
      { xRange: [t - WINDOW, t], hlines: [
        { y: p.threshold, label: 'threshold' },
        { y: vss, label: 'v_ss (steady state)' },
      ] });
    const iTop = Math.max(I, ir) * 1.1 || 1;
    figI.draw([{ x: xs, y: ring.map(() => I), label: 'I', color: 'c3', style: 'line' }],
      { xRange: [t - WINDOW, t], yRange: [0, iTop], hlines: [{ y: ir, label: 'I_rheo' }] });

    hero.narration(lifNarration({ v: state.v, p, I, splashing: splashActive(t, lastSpikeT, 0.5) }));
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${DT})])
pop = NeuronGroup(net=net, size=1, behavior=[
    LIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(1000)`;
  }

  // The live runner steps the model each frame; slider moves apply on the next
  // step without a reset (the dynamics change live).
  ui.live(figures, {
    dt: DT, unit: 'ms', modelPerSecond: MODEL_PER_SECOND, windowSize: WINDOW,
    step, reset, onFrame: frame,
  });
  // draw once so the hero is not empty before the first frame (e.g. reduced motion)
  frame();
  hero.onResize(frame);

  for (const c of [cR, cTau, cTh, cReset, cRest, cI]) c.onInput(frame);

  return { destroy() {} };
}
