import {
  HH_DEFAULTS,
  hhStep,
  hhInitialState,
  hhCurrents,
  heroLabels,
  flapAngle,
  particleRate,
  membranePolarity,
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
const V_MIN = -90, V_MAX = 60; // physiological v range, fixed on the v plot
const ION_LIFE = 0.5;    // s of real time an ion particle lives (fade-out)

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
  // Default 10 uA/cm2: periodic firing (70 Hz at dt 0.025 ms), so the plate
  // shows the action potential on load. At I = 0 nothing moves.
  const cI = ui.slider(secI, { key: 'I', label: 'I (constant)', unit: 'uA/cm2', min: 0, max: 20, step: 0.5, value: 10.0 });
  ui.button(secI, { label: 'Pulse (2 ms, +10 uA/cm2)', onClick: () => { pulseLeft = PULSE_STEPS; } });

  function params() {
    return {
      C: cC.get(), g_Na: cGN.get(), g_K: cGK.get(), g_L: cGL.get(),
      E_Na: cEN.get(), E_K: cEK.get(), E_L: cEL.get(),
      threshold: P.threshold, init_v: P.init_v,
    };
  }

  // --- live model state ---
  let state, t, ring, last, pulseLeft;
  const naParticles = [], kParticles = [];
  let naAcc = 0, kAcc = 0;
  let lastNow = 0;
  let spikeGlow = 0; // 1 on a spike, decays each frame
  // Smoothed gate values, used ONLY for the drawn flap angles (the numbers
  // shown are always the raw model sample).
  const sm = { m: 0, h: 0, n: 0 };
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
    spikeGlow = 0;
    sm.m = state.m; sm.h = state.h; sm.n = state.n;
  }

  // One dt of the model: advance HH, keep a 50 ms rolling window.
  function step() {
    const p = params();
    const I = cI.get() + (pulseLeft > 0 ? PULSE_AMP : 0);
    if (pulseLeft > 0) pulseLeft--;
    const r = hhStep(state, p, I, DT);
    state = { v: r.v, m: r.m, h: r.h, n: r.n };
    t += DT;
    last = { t, v: r.v, m: r.m, h: r.h, n: r.n, spike: r.spike };
    if (r.spike) spikeGlow = 1;
    ring.push(last);
    while (ring.length && ring[0].t < t - WINDOW) ring.shift();
  }

  // --- hero (membrane patch) ---
  const hero = ui.hero(figures, { height: 340 });

  // --- rolling plots (below the hero, linked by the shared xRange) ---
  const figV = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figG = ui.plot(figures, { height: 180, xLabel: 't', yLabel: 'gate', xUnit: 'ms', yUnit: 'a.u.' });
  const figI = ui.plot(figures, { height: 180, xLabel: 't', yLabel: 'current', xUnit: 'ms', yUnit: 'uA/cm2' });

  function redraw() {
    if (!last) return;
    const x = ring.map((s) => s.t);
    const xRange = [t - WINDOW, t];
    figV.draw([{ x, y: ring.map((s) => s.v), label: 'v', color: 'c1', style: 'line' }],
      { xRange, yRange: [V_MIN, V_MAX], hlines: [{ y: P.threshold, label: 'threshold 0 mV' }] });
    figG.draw([
      { x, y: ring.map((s) => s.m), label: 'm', color: 'c2', style: 'line' },
      { x, y: ring.map((s) => s.h), label: 'h', color: 'c4', style: 'line' },
      { x, y: ring.map((s) => s.n), label: 'n', color: 'c5', style: 'line' },
    ], { xRange, yRange: [0, 1] });
    // symmetric auto range with a minimum span of 20 uA/cm2
    let cmax = 0;
    for (const s of ring) {
      const c = hhCurrents(s, params());
      cmax = Math.max(cmax, Math.abs(c.i_Na), Math.abs(c.i_K));
    }
    const cspan = Math.max(10, cmax * 1.1);
    figI.draw([
      { x, y: ring.map((s) => hhCurrents(s, params()).i_Na), label: 'I_Na', color: 'c2', style: 'line' },
      { x, y: ring.map((s) => hhCurrents(s, params()).i_K), label: 'I_K', color: 'c4', style: 'line' },
    ], { xRange, yRange: [-cspan, cspan] });
    drawHero();
  }

  // --- hero: a lipid bilayer centred vertically, with two large Na+ pores and
  // two large K+ pores drawn as filled proteins. Flaps open by their gate
  // variable; ions flow through and disperse; the cytoplasm tints with v; a
  // voltmeter shows v; a spike glows the membrane.
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

    // membrane centred vertically: equal outside (top) and inside (bottom) bands
    const bandMid = H * 0.5;
    const yTop = bandMid - 20, yBot = bandMid + 20;

    // ease only the drawn flap angles (never the displayed numbers)
    sm.m += (last.m - sm.m) * 0.25;
    sm.h += (last.h - sm.h) * 0.25;
    sm.n += (last.n - sm.n) * 0.25;

    // cytoplasm (inside, below the band) tints with v: indigo -> neutral -> coral
    const pol = membranePolarity(last.v);
    ctx.fillStyle = pol < 0 ? pal.k : pal.na;
    ctx.globalAlpha = 0.10 + 0.28 * Math.abs(pol);
    ctx.fillRect(0, yBot, W, H - yBot);
    ctx.globalAlpha = 1;

    // spike glow along the membrane
    if (spikeGlow > 0.01) {
      ctx.strokeStyle = pal.na;
      ctx.globalAlpha = 0.5 * spikeGlow;
      ctx.lineWidth = 10;
      ctx.beginPath(); ctx.moveTo(0, bandMid); ctx.lineTo(W, bandMid); ctx.stroke();
      ctx.globalAlpha = 1;
      spikeGlow *= 0.85; // fades over ~150 ms of real time
    }

    // lipid bilayer: two rows of head circles with tails
    ctx.strokeStyle = pal.rule;
    ctx.lineWidth = 1;
    for (let x = 4; x < W; x += 11) {
      ctx.fillStyle = pal.rule;
      ctx.beginPath(); ctx.arc(x, yTop, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(x, yBot, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.moveTo(x, yTop + 3); ctx.lineTo(x, bandMid); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, yBot - 3); ctx.lineTo(x, bandMid); ctx.stroke();
    }

    // two Na pores (left half) and two K pores (right half), large
    const poreW = Math.min(70, W * 0.16), poreH = Math.min(110, H * 0.32);
    const naX = [W * 0.16, W * 0.34], kX = [W * 0.66, W * 0.84];
    const poreY = bandMid - poreH / 2;

    // advance + spawn particles (rate proportional to the real currents)
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const dtReal = lastNow ? Math.min(0.05, (now - lastNow) / 1000) : 0.016;
    lastNow = now;
    const speed = (yBot - yTop + 60) / 0.15; // px per second to cross the band
    const curNow = hhCurrents(last, params());
    stepParticles(naParticles, dtReal, speed, 1, yTop, yBot, rng);
    stepParticles(kParticles, dtReal, speed, -1, yTop, yBot, rng);
    spawn(naParticles, naX, particleRate(curNow.i_Na, 3.0) * dtReal, yTop - 24, 60, rng);
    spawn(kParticles, kX, particleRate(curNow.i_K, 3.0) * dtReal, yBot + 24, 60, rng);

    // draw pores: flap angles from the eased copy (sm), printed labels from the
    // raw model sample (last) so they match the current readout
    for (const x of naX) drawNaPore(ctx, x, poreY, poreW, poreH, sm, last, pal, mono);
    for (const x of kX) drawKPore(ctx, x, poreY, poreW, poreH, sm, last, pal, mono);

    // draw particles (fading)
    for (const p of naParticles) { ctx.fillStyle = pal.na; ctx.globalAlpha = p.alpha; dot(ctx, p.x, p.y, 3); }
    for (const p of kParticles) { ctx.fillStyle = pal.k; ctx.globalAlpha = p.alpha; dot(ctx, p.x, p.y, 3); }
    ctx.globalAlpha = 1;

    // faint ion clouds: Na+ mostly outside, K+ mostly inside (concentration gradient)
    drawCloud(ctx, W, 6, yTop - 8, pal.na, 0.25, rng);
    drawCloud(ctx, W, yBot + 8, H - 6, pal.k, 0.25, rng);

    // voltmeter: a small gauge on the right showing v
    drawVoltmeter(ctx, W - 46, H * 0.14, 30, last.v, pal, mono);

    // readouts: every printed number (gate labels + currents) comes from
    // heroLabels(last), so the m/h/n labels and the I_Na/I_K readout are the
    // same model sample and mutually consistent.
    const lab = heroLabels(last, params());
    ctx.fillStyle = pal.ink; ctx.font = mono; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`v = ${last.v.toFixed(1)} mV`, 8, 8);
    ctx.fillText(`I_Na = ${lab.i_Na.toFixed(2)} uA/cm2`, 8, 22);
    ctx.fillText(`I_K = ${lab.i_K.toFixed(2)} uA/cm2`, 8, 36);
    ctx.textAlign = 'right';
    ctx.fillText('outside (extracellular)', W - 8, 8);
    ctx.fillText('inside (cytoplasm)', W - 8, H - 18);

    hero.narration(hhNarration({ v: last.v, v_rest: P.init_v, i_Na: lab.i_Na, i_K: lab.i_K, h: last.h }));
  }

  // A Na+ pore: filled protein (two rounded half-cylinders) with 3 m flaps at
  // the outer (top) mouth and an h ball-and-chain at the inner (bottom) mouth.
  // `sm` (eased) drives the flap angles; `last` (raw) drives the printed labels.
  function drawNaPore(ctx, cx, y0, w, h, sm, last, pal, mono) {
    const half = w / 2, wallW = Math.max(14, w * 0.26);
    drawProtein(ctx, cx, y0, w, h, wallW, pal.na);
    // 3 m flaps at the outer mouth (top), hinged at the pore gap
    const mAngle = flapAngle(sm.m);
    const gapL = cx - half + wallW, gapR = cx + half - wallW;
    for (let i = 0; i < 3; i++) {
      const fx = gapL + (i + 0.5) * ((gapR - gapL) / 3);
      drawFlap(ctx, fx, y0 + 4, mAngle, pal.na);
    }
    // h ball-and-chain at the inner mouth (bottom): the ball swings into the
    // pore as h -> 0 (closed) and retracts as h -> 1 (open)
    const swing = (1 - sm.h) * (Math.PI / 2); // 0 (open) .. 90 deg (plugged)
    const chainLen = 16;
    const bx = cx + Math.sin(swing) * chainLen;
    const by = y0 + h - 4 - Math.cos(swing) * chainLen;
    ctx.strokeStyle = pal.na; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, y0 + h - 4); ctx.lineTo(bx, by); ctx.stroke();
    ctx.fillStyle = pal.na; ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.fill();
    // labels from the raw model sample (last), matching the current readout.
    // The gate value sits under the channel name on its own line (both centred
    // on the pore) so it never runs into the name or the neighbouring pore.
    ctx.fillStyle = pal.na; ctx.font = mono; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('Na+', cx, y0 - 28);
    ctx.fillText(`m=${last.m.toFixed(2)}`, cx, y0 - 14);
    ctx.fillText(`h=${last.h.toFixed(2)}`, cx, y0 + h + 6);
  }

  // A K+ pore: filled protein with 4 n flaps at the outer mouth.
  // `sm` (eased) drives the flap angles; `last` (raw) drives the printed label.
  function drawKPore(ctx, cx, y0, w, h, sm, last, pal, mono) {
    const half = w / 2, wallW = Math.max(14, w * 0.26);
    drawProtein(ctx, cx, y0, w, h, wallW, pal.k);
    const nAngle = flapAngle(sm.n);
    const gapL = cx - half + wallW, gapR = cx + half - wallW;
    for (let i = 0; i < 4; i++) {
      const fx = gapL + (i + 0.5) * ((gapR - gapL) / 4);
      drawFlap(ctx, fx, y0 + 4, nAngle, pal.k);
    }
    ctx.fillStyle = pal.k; ctx.font = mono; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`n=${last.n.toFixed(2)}`, cx, y0 - 6);
    ctx.fillText('K+', cx, y0 - 20);
  }

  // A filled protein channel: two rounded half-cylinders (left and right walls)
  // in a muted tint of `color`, with the pore gap between them.
  function drawProtein(ctx, cx, y0, w, h, wallW, color) {
    const half = w / 2;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.28;
    roundRect(ctx, cx - half, y0, wallW, h, 6); ctx.fill();
    roundRect(ctx, cx + half - wallW, y0, wallW, h, 6); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    roundRect(ctx, cx - half, y0, wallW, h, 6); ctx.stroke();
    roundRect(ctx, cx + half - wallW, y0, wallW, h, 6); ctx.stroke();
  }

  // A flap: a thick rounded paddle hinged at (x, y), closed pointing down into
  // the pore, opening by `angle` degrees to the side.
  function drawFlap(ctx, x, y, angle, color) {
    const len = 14;
    const a = (angle * Math.PI) / 180;
    ctx.strokeStyle = color; ctx.lineWidth = 6; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.sin(a) * len, y + Math.cos(a) * len);
    ctx.stroke();
    ctx.lineCap = 'butt';
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function dot(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }

  // Advance particles: transit straight through the band, then disperse (random
  // walk) and fade over ION_LIFE seconds. dir +1 = outside->inside (Na), -1 =
  // inside->outside (K).
  function stepParticles(arr, dtReal, speed, dir, yTop, yBot, rng) {
    for (let i = arr.length - 1; i >= 0; i--) {
      const p = arr[i];
      p.age += dtReal;
      p.alpha = Math.max(0, 1 - p.age / ION_LIFE);
      if (p.alpha <= 0) { arr.splice(i, 1); continue; }
      const crossed = dir > 0 ? p.y > yBot : p.y < yTop;
      if (!crossed) {
        p.y += dir * speed * dtReal;
      } else {
        // disperse into the compartment
        p.x += (rng.next() - 0.5) * 6;
        p.y += (rng.next() - 0.5) * 6;
      }
    }
  }

  // Spawn floor(acc) particles at the given pore x positions, keep the fraction.
  function spawn(arr, xs, expected, yStart, cap, rng) {
    if (expected <= 0) return;
    arr._acc = (arr._acc || 0) + expected;
    const count = Math.floor(arr._acc);
    arr._acc -= count;
    for (let i = 0; i < count && arr.length < cap * xs.length; i++) {
      const x = xs[Math.floor(rng.next() * xs.length)] + (rng.next() - 0.5) * 12;
      arr.push({ x, y: yStart, age: 0, alpha: 1 });
    }
  }

  // A faint, static cloud of ions in a compartment band (decorative).
  function drawCloud(ctx, W, y0, y1, color, alpha, rng) {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    const n = Math.floor(W / 22);
    for (let i = 0; i < n; i++) {
      const x = (i * 97.3 + 13) % W;
      const y = y0 + ((i * 53.7 + 7) % Math.max(1, y1 - y0));
      dot(ctx, x, y, 1.6);
    }
    ctx.globalAlpha = 1;
  }

  // A small voltmeter: a semicircle gauge with a needle pointing at v.
  function drawVoltmeter(ctx, cx, cy, r, v, pal, mono) {
    ctx.strokeStyle = pal.rule; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI); ctx.stroke();
    const f = Math.max(0, Math.min(1, (v - V_MIN) / (V_MAX - V_MIN)));
    const a = Math.PI + f * Math.PI;
    ctx.strokeStyle = pal.v; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * (r - 4), cy + Math.sin(a) * (r - 4));
    ctx.stroke();
    ctx.fillStyle = pal.ink; ctx.font = mono; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`${v.toFixed(0)} mV`, cx, cy + 4);
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
  crossings of the threshold (default 0 mV); there is no reset. The default I = 10 uA/cm2 drives
  periodic firing (70 Hz at dt = 0.025 ms).`);
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
