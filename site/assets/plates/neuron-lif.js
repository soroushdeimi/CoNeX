import {
  DEFAULTS,
  lifStep,
  lifRheobase,
  bucketFlows,
  bucketLevel,
  splashActive,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-lif',
  title: 'Leaky integrate-and-fire',
  conex: ['LIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 500;

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- controls ---
  const sec = ui.section(controls, 'Model parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });

  const secI = ui.section(controls, 'Input current');
  const cI = ui.slider(secI, { key: 'I', label: 'I (constant)', unit: 'pA', min: 0, max: 120, step: 1, value: 65.0 });

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(),
      v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get(),
    };
  }

  // Compute the whole run once; the player reveals it frame by frame.
  let runData = null;
  function computeRun() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], i = [], spikes = [];
    let state = { v: p.v_rest };
    for (let n = 0; n < STEPS; n++) {
      const r = lifStep(state, p, I, p.dt);
      state = { v: r.v };
      t.push(n * p.dt);
      v.push(r.v);
      i.push(I);
      spikes.push(r.spike);
    }
    const ir = lifRheobase(p);
    const vss = p.v_rest + p.R * I;
    const spikeTimes = [];
    for (let n = 0; n < spikes.length; n++) if (spikes[n]) spikeTimes.push(n * p.dt);
    runData = { p, I, t, v, i, spikes, spikeTimes, ir, vss };
    return t[t.length - 1];
  }

  // --- figures ---
  const fig = ui.plot(figures, { height: 260, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figI = ui.plot(figures, { height: 120, xLabel: 't', yLabel: 'I', xUnit: 'ms', yUnit: 'pA' });
  // Leaky-bucket picture of the LIF (the main figure; it replaces the old
  // membrane gauge), driven by the same run and the same t.
  const bucket = ui.canvas(figures, { height: 344 });
  const bucketCap = document.createElement('p');
  bucketCap.style.fontStyle = 'italic';
  bucketCap.style.color = 'var(--muted)';
  bucketCap.style.fontSize = '0.85rem';
  bucketCap.textContent =
    'Analogy: water level = membrane potential, tap = input current, hole = leak through the ' +
    'membrane (R, tau). Where it breaks: a real hole leaks in proportion to the square root of ' +
    'the height; the LIF leak is linear in v − v_rest. The neuron does not spill over: CoNeX ' +
    'resets v to v_reset when v ≥ threshold.';
  figures.appendChild(bucketCap);
  // Build the run before the player: ui.player may call onFrame synchronously.
  const player = ui.player(figures, { duration: computeRun(), unit: 'ms', playSeconds: 5, loop: true, onFrame: frame });

  // --- notes ---
  ui.note(notes, `The LIF neuron integrates its input current and leaks back to rest. CoNeX updates the
  voltage once per iteration with the Euler step below, then <code>Fire</code> resets any neuron that
  reached the threshold. CoNeX has no refractory period: the reset is instantaneous and the neuron can
  fire again on the very next step. Units: v in mV, t in ms, I in pA, R in GOhm (so R·I is in mV).`);
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

  // Reveal the traces up to time t and draw the membrane gauge at the real v(t).
  function frame(t) {
    if (!runData) return;
    const { p, I, t: tt, v, i, ir, vss } = runData;
    fig.draw(
      [{ x: tt, y: v, label: 'v', color: 'c1', style: 'line' }],
      { revealX: t, hlines: [
        { y: p.threshold, label: 'threshold' },
        { y: vss, label: 'v_ss (steady state)' },
      ] }
    );
    // Constant input: pin the axis to start at 0 so the value is readable.
    const iTop = Math.max(I, ir) * 1.1 || 1;
    figI.draw([{ x: tt, y: i, label: 'I', color: 'c3', style: 'line' }],
      { revealX: t, yRange: [0, iTop], hlines: [{ y: ir, label: 'I_rheo' }] });
    drawBucket(t);
  }

  // Leaky-bucket picture — the main figure. Water level = the REAL v(t) at the
  // player time, always (it drops to v_reset at the spike step, which is the real
  // reset). Tap inflow = R*I/tau; hole outflow = (v - v_rest)/tau (bucketFlows).
  // A spike adds only a brief coral tint + droplets over the rim for 0.5 ms of
  // model time (splashActive); it never holds the level at the rim.
  function drawBucket(t) {
    const { p, I, v, spikeTimes } = runData;
    const idx = Math.min(v.length - 1, Math.max(0, Math.round(t / p.dt)));
    const vNow = v[idx];
    const { input, leak } = bucketFlows({ R: p.R, I, tau: p.tau, v_rest: p.v_rest }, vNow);

    // Splash: only a brief tint + droplets within 0.5 ms of model time after the
    // most recent spike. The level itself is always the real v(t).
    let lastSpikeT = null;
    for (const st of spikeTimes) if (st <= t) lastSpikeT = st;
    const splashing = splashActive(t, lastSpikeT, 0.5);

    const ctx = bucket.ctx;
    const W = bucket.width, H = bucket.height;
    ctx.clearRect(0, 0, W, H);
    const pal = {
      ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'),
      water: ui.color('c1'), tap: ui.color('c3'), leak: ui.color('c4'), coral: ui.color('c2'),
    };
    const mono = '11px ui-monospace, SFMono-Regular, Menlo, monospace';

    // Bucket geometry: centred, sized to fill the width sensibly. (The min width
    // is kept low enough that the left-wall level labels fit at 390 px.)
    const bw = Math.min(240, Math.max(128, W * 0.36));
    const bx = (W - bw) / 2;
    const cx = bx + bw / 2;
    const yRim = 90, yBot = 280;            // rim (threshold) and floor (v_rest)
    const yOf = (val) => yBot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (yBot - yRim);

    // Water fill from v_rest up to the REAL current level (bucketLevel of v(t)).
    const level = bucketLevel(vNow, p);
    const yWater = yBot - level * (yBot - yRim);
    ctx.fillStyle = splashing ? pal.coral : pal.water;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(bx + 2, yWater, bw - 4, yBot - yWater);
    ctx.globalAlpha = 1;
    // light surface line with a subtle ripple
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = bx + 2; x <= bx + bw - 2; x += 4) {
      const y = yWater + Math.sin((x - bx) / 9) * 1.2;
      if (x === bx + 2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glass bucket: open top, faint walls.
    ctx.strokeStyle = pal.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, yRim); ctx.lineTo(bx, yBot); ctx.lineTo(bx + bw, yBot); ctx.lineTo(bx + bw, yRim);
    ctx.stroke();

    // Dashed reference lines: threshold (rim) and v_reset. The level labels sit
    // on the LEFT wall (right-aligned) so the right side is free for the leak jet.
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

    // Tap above the bucket; falling stream width = input.
    const tapW = Math.max(2, Math.min(30, 2 + input * 5));
    ctx.strokeStyle = pal.tap;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(cx - 18, 40); ctx.lineTo(cx + 18, 40); ctx.stroke(); // pipe
    ctx.lineWidth = tapW;
    ctx.beginPath(); ctx.moveTo(cx, 44); ctx.lineTo(cx, yWater - 2); ctx.stroke();   // stream
    ctx.fillStyle = pal.tap;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`input R·I/τ = ${input.toFixed(2)} mV/ms`, cx, 34);

    // Hole near the bottom of the right wall; arcing jet width/length = leak.
    const holeY = yBot - 20;
    const leakW = Math.max(1.5, Math.min(30, 1.5 + leak * 5));
    const jetLen = Math.max(10, Math.min(70, 10 + leak * 40));
    ctx.strokeStyle = pal.leak;
    ctx.lineWidth = leakW;
    ctx.beginPath();
    ctx.moveTo(bx + bw - 2, holeY);
    ctx.quadraticCurveTo(bx + bw + jetLen * 0.5, holeY + 4, bx + bw + jetLen, holeY + jetLen * 0.5);
    ctx.stroke();
    // Leak label: centred below the bucket floor (below the jet's end), so it
    // never collides with the v_rest / v_reset / threshold marks on the right.
    ctx.fillStyle = pal.leak;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`leak (v − v_rest)/τ = ${leak.toFixed(2)} mV/ms`, cx, yBot + 14);

    // v readout on the water surface (left of the tap stream, which falls at centre).
    ctx.fillStyle = pal.ink;
    ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`v = ${vNow.toFixed(1)} mV`, bx + 8, yWater - 4);

    // Splash: droplets over the rim while splashing.
    if (splashing) {
      ctx.fillStyle = pal.coral;
      for (const [dx, dy] of [[-24, -14], [-8, -22], [8, -20], [24, -12], [34, -6]]) {
        ctx.beginPath(); ctx.arc(cx + dx, yRim + dy, 3, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Spike strip: a tick at each real spike time up to t.
    const tMax = v.length * p.dt;
    const sx0 = bx, sx1 = bx + bw, sy = 328, sh = 16;
    ctx.strokeStyle = pal.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(sx0, sy, sx1 - sx0, sh);
    ctx.fillStyle = pal.ink;
    for (const st of spikeTimes) {
      if (st > t) continue;
      const px = sx0 + (st / tMax) * (sx1 - sx0);
      ctx.fillRect(px - 0.75, sy + 2, 1.5, sh - 4);
    }
    ctx.fillStyle = pal.muted;
    ctx.font = mono;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('spikes', sx0, sy - 3);
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=1, behavior=[
    LIF(R=${p.R}, threshold=${p.threshold}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    Fire(),
])
net.initialize()
pop.I = pop.vector(mode=${cI.get()})   # constant input current (pA)
net.simulate_iterations(${STEPS})`;
  }

  // On a parameter change: recompute the run and continue from the current t (clamped).
  function run() {
    player.setDuration(computeRun());
  }

  for (const c of [cR, cTau, cTh, cReset, cRest, cDt, cI]) c.onInput(run);
  ui.onTheme(run);

  return { destroy() {} };
}
