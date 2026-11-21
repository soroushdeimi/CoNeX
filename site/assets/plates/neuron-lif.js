import {
  DEFAULTS,
  lifStep,
  lifRheobase,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-lif',
  title: 'Leaky integrate-and-fire',
  conex: ['LIF'],
  source: 'conex/behaviors/neurons/neuron_types/lif_neurons.py',
};

const P = { ...DEFAULTS };
const STEPS = 500;

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

  // --- figures ---
  const fig = ui.plot(figures, { height: 260, xLabel: 't', yLabel: 'v', xUnit: 'ms', yUnit: 'mV' });
  const figI = ui.plot(figures, { height: 120, xLabel: 't', yLabel: 'I', xUnit: 'ms', yUnit: 'pA' });
  // Membrane gauge: a vertical bar from v_rest to threshold filled to the current v.
  const gauge = ui.canvas(figures, { height: 260 });
  const player = makePlayer(ui, figures, frame);

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

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(),
      v_reset: cReset.get(), v_rest: cRest.get(), dt: cDt.get(),
    };
  }

  let runData = null;

  // Recompute the whole run once; the player then reveals it frame by frame.
  function run() {
    const p = params();
    const I = cI.get();
    const t = [], v = [], i = [];
    let state = { v: p.v_rest };
    for (let n = 0; n < STEPS; n++) {
      const r = lifStep(state, p, I, p.dt);
      state = { v: r.v };
      t.push(n * p.dt);
      v.push(r.v);
      i.push(I);
    }
    const ir = lifRheobase(p);
    const vss = p.v_rest + p.R * I;
    runData = { p, I, t, v, i, ir, vss };
    player.setDuration(t[t.length - 1]);
    player.seek(0); // reveals the first frame (also covers reduced-motion, which starts paused)
  }

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
    drawGauge(t);
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
    const pal = { ink: ui.color('ink'), muted: ui.color('muted'), rule: ui.color('rule'), coral: ui.color('c2') };
    const x = W / 2, bw = 26, top = 14, bot = H - 14;
    const yOf = (val) => bot - ((val - p.v_rest) / (p.threshold - p.v_rest)) * (bot - top);
    // track
    ctx.strokeStyle = pal.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - bw / 2, top, bw, bot - top);
    // fill from v_rest up to the current v (clipped to the track)
    const vClamped = Math.max(p.v_rest, Math.min(p.threshold, vNow));
    const yTop = yOf(vClamped);
    ctx.fillStyle = spiked ? pal.coral : ui.color('c1');
    ctx.fillRect(x - bw / 2 + 1, yTop, bw - 2, bot - yTop);
    // threshold and rest markers
    ctx.strokeStyle = pal.muted;
    ctx.setLineDash([3, 3]);
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

  for (const c of [cR, cTau, cTh, cReset, cRest, cDt, cI]) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
