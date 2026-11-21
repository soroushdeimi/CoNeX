// Plate: Spike-timing dependent plasticity (SimpleSTDP).
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'syn-stdp',
  title: 'Spike-timing dependent plasticity',
  conex: ['SimpleSTDP', 'PreTrace', 'PostTrace'],
  source: 'conex/behaviors/synapses/learning.py',
};

// Simulate the pre trace after a single pre spike at t=0.
function preTraceAfterSpike(tau_s, dt, steps) {
  let tr = 0;
  const out = [0];
  for (let t = 0; t < steps; t++) {
    const spike = t === 0 ? 1 : 0;
    tr = M.stepTrace(tr, spike, { tau_s, dt });
    out.push(tr);
  }
  return out; // out[t] = trace after step t
}

// STDP window: dw for a single pre spike and a single post spike separated by delta_t.
function stdpWindow({ a_plus, a_minus, tau_s, dt, bound, w_min, w_max, w }) {
  const maxT = 60;
  const pre = preTraceAfterSpike(tau_s, dt, maxT);
  const post = preTraceAfterSpike(tau_s, dt, maxT);
  const b = M.BOUNDS[bound] || M.noBound;
  const pts = [];
  for (let d = -maxT; d <= maxT; d++) {
    let dw = 0;
    if (d >= 0) dw = pre[d] * 1 * a_plus * b(w, w_min, w_max); // pre at 0, post at d
    else dw = -1 * post[-d] * a_minus * b(w, w_min, w_max); // post at 0, pre at -d
    pts.push([d, dw]);
  }
  return pts;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const state = {
    a_plus: 0.01, a_minus: 0.002, tau_s: 10, dt: 1,
    w_min: 0, w_max: 1, w0: 0.5, bound: 'no_bound', offset: 5, steps: 40,
  };

  const windowPlot = ui.plot(figures, { height: 200, xLabel: 'delta t', xUnit: 'ms', yLabel: 'dw', yUnit: 'a.u.' });
  const tracePlot = ui.plot(figures, { height: 200, xLabel: 'time', xUnit: 'ms', yLabel: 'trace / weight', yUnit: 'a.u.' });

  function draw() {
    const win = stdpWindow({ ...state, w: state.w0 });
    windowPlot.draw([{ x: win.map((p) => p[0]), y: win.map((p) => p[1]), label: 'STDP window', color: 'c1' }], { hlines: [{ y: 0, label: '0' }] });
    // traces + weight for a pre train and a post train offset by `offset` steps
    const T = state.steps;
    const preS = [], postS = [];
    for (let t = 0; t < T; t++) { preS.push(t % 10 === 0 ? 1 : 0); postS.push((t - state.offset) % 10 === 0 && t >= state.offset ? 1 : 0); }
    let preT = 0, postT = 0, w = state.w0;
    const xs = [], preTs = [], postTs = [], ws = [];
    for (let t = 0; t < T; t++) {
      preT = M.stepTrace(preT, preS[t], { tau_s: state.tau_s, dt: state.dt });
      postT = M.stepTrace(postT, postS[t], { tau_s: state.tau_s, dt: state.dt });
      const dw = M.stdpDw({ weights: [w], pre_spike: [preS[t]], post_spike: [postS[t]], pre_trace: [preT], post_trace: [postT], a_plus: state.a_plus, a_minus: state.a_minus, w_min: state.w_min, w_max: state.w_max, positive_bound: state.bound, negative_bound: state.bound })[0];
      w += dw;
      xs.push(t); preTs.push(preT); postTs.push(postT); ws.push(w);
    }
    tracePlot.draw([
      { x: xs, y: preTs, label: 'pre_trace', color: 'c1' },
      { x: xs, y: postTs, label: 'post_trace', color: 'c2' },
      { x: xs, y: ws, label: 'weight', color: 'c4' },
    ], { hlines: [{ y: state.w_min, label: 'w_min' }, { y: state.w_max, label: 'w_max' }] });
  }

  const sec = ui.section(controls, 'STDP rule');
  ui.slider(sec, { key: 'a_plus', label: 'a_plus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_plus, onInput: (v) => { state.a_plus = v; draw(); } });
  ui.slider(sec, { key: 'a_minus', label: 'a_minus', unit: 'a.u.', min: 0, max: 0.05, step: 0.001, value: state.a_minus, onInput: (v) => { state.a_minus = v; draw(); } });
  ui.slider(sec, { key: 'tau_s', label: 'tau_s', unit: 'ms', min: 1, max: 50, step: 1, value: state.tau_s, onInput: (v) => { state.tau_s = v; draw(); } });
  ui.slider(sec, { key: 'w0', label: 'w0', unit: 'a.u.', min: 0, max: 1, step: 0.05, value: state.w0, onInput: (v) => { state.w0 = v; draw(); } });
  ui.slider(sec, { key: 'offset', label: 'post offset', unit: 'steps', min: -10, max: 10, step: 1, value: state.offset, onInput: (v) => { state.offset = v; draw(); } });
  ui.select(sec, { key: 'bound', label: 'bound', value: state.bound, options: [{ value: 'no_bound', label: 'no_bound' }, { value: 'hard_bound', label: 'hard_bound' }, { value: 'soft_bound', label: 'soft_bound' }], onInput: (v) => { state.bound = v; draw(); } });

  ui.onTheme(draw);
  draw();

  ui.note(notes, `STDP changes a synapse weight depending on the timing of pre- and post-synaptic spikes. If the pre neuron fires just before the post neuron (causal pairing) the weight is potentiated; if the post fires first (anti-causal) it is depressed. This is the canonical associative learning rule for spiking networks and is thought to underlie Hebbian learning in cortex.`);
  ui.equation(notes, `<var>dw</var> = <var>a</var><sub>+</sub>·<var>pre_trace</var>·<var>post_spike</var> − <var>a</var><sub>−</sub>·<var>pre_spike</var>·<var>post_trace</var>`);
  ui.note(notes, `The traces decay with time constant tau_s (ms); the weight is dimensionless (a.u.). The window plot simulates single spike pairs to show dw versus the timing offset. ML analogy: this is like a Hebbian "fire together, wire together" update, but gated by precise timing. The analogy breaks because classical Hebbian learning uses rates, not spike timing, and has no depression term.`);
  ui.code(notes, () => `from conex import SimpleSTDP, PreTrace, PostTrace, PreSpikeCatcher, PostSpikeCatcher
SimpleSTDP(a_plus=${state.a_plus}, a_minus=${state.a_minus}, w_min=${state.w_min}, w_max=${state.w_max},
           positive_bound="${state.bound}", negative_bound="${state.bound}")
PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})`);
  return { destroy() {} };
}
