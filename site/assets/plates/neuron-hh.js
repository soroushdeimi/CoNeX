import {
  HH_DEFAULTS,
  hhStep,
  hhInitialState,
  openGates,
  particleRate,
  membraneTint,
  hhNarration,
} from '../js/models/hh.js';

export const meta = {
  id: 'neuron-hh',
  title: 'Hodgkin-Huxley (1952)',
  conex: ['HH'],
  source: 'conex/behaviors/neurons/neuron_types/hh_neurons.py',
};

const DT = 0.025;        // ms (fixed; CoNeX HH is stable at this step)
const WINDOW = 50;       // ms of model time kept on screen
const PULSE_STEPS = Math.round(2.0 / DT); // 2 ms pulse
const PULSE_AMP = 10.0;  // uA/cm2, added on top of the constant I

const P = { ...HH_DEFAULTS };

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- controls ---
  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cC = ui.slider(sec, { key: 'C', label: 'C', unit: 'uF/cm2', min: 0.1, max: 5, step: 0.1, value: P.C });
  const cGN = ui.slider(sec, { key: 'g_Na', label: 'g_Na', unit: 'mS/cm2', min: 0, max: 200, step: 1, value: P.g_Na });
  const cGK = ui.slider(sec, { key: 'g_K', label: 'g_K', unit: 'mS/cm2', min: 0, max: 100, step: 1, value: P.g_K });
  const cGL = ui.slider(sec, { key: 'g_L', label: 'g_L', unit: 'mS/cm2', min: 0, max: 2, step: 0.05, value: P.g_L });
  const cEN = ui.slider(sec, { key: 'E_Na', label: 'E_Na', unit: 'mV', min: 0, max: 100, step: 1, value: P.E_Na });
  const cEK = ui.slider(sec, { key: 'E_K', label: 'E_K', unit: 'mV', min: -100, max: -50, step: 1, value: P.E_K });
  const cEL = ui.slider(sec, { key: 'E_L', label: 'E_L', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.E_L });

  const secI = ui.section(controls, 'Input current');
  const cI = ui.slider(secI, { key: 'I', label: 'I (constant)', unit: 'uA/cm2', min: 0, max: 20, step: 0.5, value: 0.0 });
  const pulseBtn = ui.button(secI, { label: 'Pulse (2 ms, +10 uA/cm2)', onClick: () => { pulseLeft = PULSE_STEPS; } });

  function params() {
    return {
      C: cC.get(), g_Na: cGN.get(), g_K: cGK.get(), g_L: cGL.get(),
      E_Na: cEN.get(), E_K: cEK.get(), E_L: cEL.get(),
      threshold: P.threshold, init_v: P.init_v,
    };
  }

  // --- live model state ---
  let state, t, ring, last, pulseLeft;
  const naParticles = [];
  const kParticles = [];
  let naAcc = 0, kAcc = 0;
  let lastNow = 0;
  const rng = ui.rng(1234);

  function reset() {
    state = hhInitialState(params());
    t = 0;
    ring = [];
    last = null;
    pulseLeft = 0;
    naParticles.length = 0;
    kParticles.length = 0;
    naAcc = 0; kAcc = 0;
  }

  // One dt of the model: advance HH, keep a 50 ms rolling window.
  function step() {
    const p = params();
    const I = cI.get() + (pulseLeft > 0 ? PULSE_AMP : 0);
    if (pulseLeft > 0) pulseLeft--;
    const r = hhStep(state, p, I, DT);
    state = { v: r.v, m: r.m, h: r.h, n: r.n };
    t += DT;
    last = { t, v: r.v, m: r.m, h: r.h, n: r.n, i_Na: r.i_Na, i_K: r.i_K };
    ring.push(last);
    while (ring.length && ring[0].t < t - WINDOW) ring.shift();
  }

  // --- hero (membrane patch) ---
  const hero = ui.hero(figures, { height: 360 });

  // --- rolling plots (below the hero, linked by the shared xRange) ---
  const figV = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figG = ui.plot(figures, { height: 180, xLabel: 't', yLabel: 'gate', xUnit: 'ms', yUnit: 'a.u.' });
  const figI = ui.plot(figures, { height: 180, xLabel: 't', yLabel: 'current', xUnit: 'ms', yUnit: 'uA/cm2' });

  function redraw() {
    if (!last) return;
    const x = ring.map((s) => s.t);
    const xRange = [t - WINDOW, t];
    figV.draw([{ x, y: ring.map((s) => s.v), label: 'v', color: 'c1', style: 'line' }],
      { xRange, hlines: [{ y: P.threshold, label: 'threshold' }] });
    figG.draw([
      { x, y: ring.map((s) => s.m), label: 'm', color: 'c2', style: 'line' },
      { x, y: ring.map((s) => s.h), label: 'h', color: 'c4', style: 'line' },
      { x, y: ring.map((s) => s.n), label: 'n', color: 'c5', style: 'line' },
    ], { xRange, yRange: [0, 1] });
    figI.draw([
      { x, y: ring.map((s) => s.i_Na), label: 'I_Na', color: 'c2', style: 'line' },
      { x, y: ring.map((s) => s.i_K), label: 'I_K', color: 'c4', style: 'line' },
    ], { xRange });
    drawHero();
  }

  // Membrane patch: a lipid bilayer band with Na+ channels (3 m + 1 h gates)
  // and K+ channels (4 n gates). Open gates follow the real m, h, n; ions flow
  // as particles at a rate proportional to the real I_Na / I_K; the cytoplasm
  // tints with v.
  function drawHero() {
    if (!last) return;
    const { ctx } = hero;
    const W = hero.width, H = hero.height;
    ctx.clearRect(0, 0, W, H);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      na: ui.color('c2'), k: ui.color('c4'), v: ui.color('c1'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';

    const yTop = H * 0.40, yBot = H * 0.60; // bilayer band
    // cytoplasm (inside, below the band) tints with v
    const tint = membraneTint(last.v, -80, 40);
    ctx.fillStyle = pal.v;
    ctx.globalAlpha = 0.06 + 0.30 * tint;
    ctx.fillRect(0, yBot, W, H - yBot);
    ctx.globalAlpha = 1;

    // bilayer: two rows of lipid heads
    ctx.fillStyle = pal.rule;
    for (let x = 4; x < W; x += 12) {
      ctx.beginPath(); ctx.arc(x, yTop, 3, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(x, yBot, 3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = pal.muted;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, (yTop + yBot) / 2); ctx.lineTo(W, (yTop + yBot) / 2); ctx.stroke();

    // channel x positions: Na at even, K at odd
    const xs = [0.12, 0.28, 0.44, 0.60, 0.76, 0.92].map((f) => f * W);
    const isNa = [true, false, true, false, true, false];

    // advance + spawn particles (rate proportional to the real currents)
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const dtReal = lastNow ? Math.min(0.05, (now - lastNow) / 1000) : 0.016;
    lastNow = now;
    const speed = (yBot - yTop) / 0.35; // px per second to cross the band
    stepParticles(naParticles, dtReal, speed, 1);
    stepParticles(kParticles, dtReal, speed, -1);
    spawn(naParticles, xs, isNa, true, particleRate(last.i_Na, 2.0) * dtReal, yTop, yBot);
    spawn(kParticles, xs, isNa, false, particleRate(last.i_K, 2.0) * dtReal, yTop, yBot);

    // draw channels
    for (let c = 0; c < xs.length; c++) {
      const x = xs[c];
      ctx.strokeStyle = pal.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 9, yTop - 6, 18, yBot - yTop + 12);
      if (isNa[c]) {
        // 3 m gates (top) + 1 h gate (bottom)
        drawGates(ctx, x, yTop, yBot, openGates(last.m, 3), 3, pal.na);
        drawGates(ctx, x, yBot - 14, yBot, openGates(last.h, 1), 1, pal.na);
        ctx.fillStyle = pal.na; ctx.font = mono; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText('Na', x, yTop - 8);
      } else {
        drawGates(ctx, x, yTop, yBot, openGates(last.n, 4), 4, pal.k);
        ctx.fillStyle = pal.k; ctx.font = mono; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText('K', x, yTop - 8);
      }
    }

    // draw particles
    for (const p of naParticles) { ctx.fillStyle = pal.na; dot(ctx, p.x, p.y, 2.5); }
    for (const p of kParticles) { ctx.fillStyle = pal.k; dot(ctx, p.x, p.y, 2.5); }

    // readouts
    ctx.fillStyle = pal.ink; ctx.font = mono; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`v = ${last.v.toFixed(1)} mV`, 8, 8);
    ctx.fillText(`I_Na = ${last.i_Na.toFixed(2)} uA/cm2`, 8, 22);
    ctx.fillText(`I_K = ${last.i_K.toFixed(2)} uA/cm2`, 8, 36);
    ctx.textAlign = 'right';
    ctx.fillText('outside (extracellular)', W - 8, 8);
    ctx.fillText('inside (cytoplasm)', W - 8, H - 18);

    hero.narration(hhNarration({ v: last.v, v_rest: P.init_v, i_Na: last.i_Na, i_K: last.i_K, h: last.h }));
  }

  function drawGates(ctx, x, y0, y1, openCount, total, color) {
    const gap = (y1 - y0) / (total + 1);
    for (let i = 0; i < total; i++) {
      const y = y0 + gap * (i + 1);
      const open = i < openCount;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      if (open) { ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke(); }
      else { ctx.fillStyle = color; ctx.fill(); }
    }
  }

  function dot(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }

  function stepParticles(arr, dtReal, speed, dir) {
    for (let i = arr.length - 1; i >= 0; i--) {
      arr[i].y += dir * speed * dtReal;
      if (dir > 0 && arr[i].y > yBotOf()) arr.splice(i, 1);
      if (dir < 0 && arr[i].y < yTopOf()) arr.splice(i, 1);
    }
  }
  function yTopOf() { return hero.height * 0.40; }
  function yBotOf() { return hero.height * 0.60; }

  // Spawn floor(acc) particles at open-channel x positions, keep the fraction.
  function spawn(arr, xs, isNa, wantNa, expected, yTop, yBot) {
    naAcc += wantNa ? expected : 0;
    kAcc += wantNa ? 0 : expected;
    const acc = wantNa ? naAcc : kAcc;
    const count = Math.floor(acc);
    if (wantNa) naAcc -= count; else kAcc -= count;
    const channels = xs.map((x, i) => (wantNa ? isNa[i] : !isNa[i]) ? x : null).filter((x) => x != null);
    if (!channels.length) return;
    for (let i = 0; i < count && arr.length < 200; i++) {
      const x = channels[Math.floor(rng.next() * channels.length)] + (rng.next() - 0.5) * 6;
      arr.push({ x, y: wantNa ? yTop : yBot });
    }
  }

  // --- notes ---
  ui.note(notes, `The Hodgkin-Huxley model (1952) describes the action potential of the squid giant
  axon. The membrane potential is driven by three voltage-dependent conductances: a fast sodium
  current (gated by m and h), a delayed potassium current (gated by n), and a leak. CoNeX integrates
  all four ODEs with forward Euler at the network's dt. Units: v in mV, t in ms, I in uA/cm2,
  C in uF/cm2, g in mS/cm2, E in mV.`);
  ui.equation(notes, `<var>C</var> d<var>v</var>/d<var>t</var> = <var>I</var> − <var>g</var><sub>Na</sub><var>m</var><sup>3</sup><var>h</var>(<var>v</var>−<var>E</var><sub>Na</sub>) − <var>g</var><sub>K</sub><var>n</var><sup>4</sup>(<var>v</var>−<var>E</var><sub>K</sub>) − <var>g</var><sub>L</sub>(<var>v</var>−<var>E</var><sub>L</sub>)`);
  ui.equation(notes, `d<var>x</var>/d<var>t</var> = <var>alpha</var><sub>x</sub>(1−<var>x</var>) − <var>beta</var><sub>x</sub><var>x</var>,  <var>x</var> ∈ {<var>m</var>, <var>h</var>, <var>n</var>}`);
  ui.note(notes, `The gate rates are the Hodgkin &amp; Huxley (1952) fit at 6.3 degC in the modern
  absolute-voltage convention (rest about −65 mV). The constants g_Na=120, g_K=36, g_L=0.3 mS/cm2,
  E_Na=50, E_K=−77, E_L=−54.387 mV, C=1 uF/cm2 are the classic squid-axon values. Spikes are upward
  crossings of the threshold (default 0 mV); there is no reset.`);
  ui.note(notes, `ML analogy: the three gates are like the gated units of a GRU — each is a
  voltage-dependent "sigmoid" with its own time constant that opens or closes a conductance. The
  analogy breaks because these gates are not learned sigmoids: they are fixed, voltage-dependent
  kinetics (alpha/beta rate functions) with physical time constants, not trainable weights.`);
  ui.note(notes, `Honest numerics: CoNeX uses forward Euler, so spike counts depend on dt — at
  10 uA/cm2 and dt=0.025 ms the neuron fires 14 spikes in 200 ms (70 Hz). An exact-exponential gate
  update (as in some references) differs from CoNeX's Euler gates by about 0.7 mV over 50 ms; the
  difference is the gate integrator, not the constants.`);
  ui.code(notes, () => codeSnippet());

  function codeSnippet() {
    const p = params();
    return `import os, sys
# The editable CoNeX install points at the main checkout; this worktree has the
# HH behavior, so put it (three levels up from this snippet) on the path first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.hh_neurons import HH
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${DT})], dtype=torch.float64)
pop = NeuronGroup(net=net, size=1, behavior=[
    HH(C=${p.C}, g_Na=${p.g_Na}, g_K=${p.g_K}, g_L=${p.g_L},
       E_Na=${p.E_Na}, E_K=${p.E_K}, E_L=${p.E_L},
       threshold=${P.threshold}, init_v=${P.init_v}),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (uA/cm2)
net.simulate_iterations(2000)   # 50 ms
print(f"v = {pop.v.item():.3f} mV")`;
  }

  // Initialize the model state ourselves: the real ui.live does not call
  // reset() on mount (only the Reset button does), so step() must never run
  // before reset() has set `state`.
  reset();

  // The live runner steps the model each frame; the plots scroll a 50 ms window.
  ui.live(figures, {
    dt: DT, unit: 'ms', modelPerSecond: 20, windowSize: WINDOW,
    step, reset, onFrame: redraw,
  });
  // fill the window once so the figures are not empty before the first frame
  for (let i = 0; i < WINDOW / DT; i++) step();
  redraw();

  hero.onResize(redraw);
  ui.onTheme(redraw);
  for (const c of [cC, cGN, cGK, cGL, cEN, cEK, cEL, cI]) c.onInput(redraw);

  return { destroy() {} };
}
