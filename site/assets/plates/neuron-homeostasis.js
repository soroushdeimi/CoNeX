import {
  DEFAULTS,
  lifStep,
  makeActivityHomeostasis,
  makeVoltageHomeostasis,
} from '../js/models/neurons.js';

export const meta = {
  id: 'neuron-homeostasis',
  title: 'Homeostatic threshold control',
  conex: ['ActivityBaseHomeostasis', 'VoltageBaseHomeostasis'],
  source: 'conex/behaviors/neurons/homeostasis.py',
};

const P = { ...DEFAULTS };
const SIZE = 8;
const STEPS = 50000; // 5 s at dt = 0.1 ms, long enough to converge

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Neuron (LIF) parameters (CoNeX names)');
  const cR = ui.slider(sec, { key: 'R', label: 'R', unit: 'GOhm', min: 0.1, max: 2, step: 0.05, value: P.R });
  const cTau = ui.slider(sec, { key: 'tau', label: 'tau', unit: 'ms', min: 1, max: 50, step: 0.5, value: P.tau });
  const cTh = ui.slider(sec, { key: 'threshold', label: 'threshold (init)', unit: 'mV', min: -80, max: 20, step: 0.5, value: -50.0 });
  const cReset = ui.slider(sec, { key: 'v_reset', label: 'v_reset', unit: 'mV', min: -80, max: -40, step: 0.5, value: P.v_reset });
  const cRest = ui.slider(sec, { key: 'v_rest', label: 'v_rest', unit: 'mV', min: -80, max: -50, step: 0.5, value: P.v_rest });
  const cDt = ui.slider(sec, { key: 'dt', label: 'dt', unit: 'ms', min: 0.01, max: 1, step: 0.01, value: P.dt });

  const secIn = ui.section(controls, 'Random input (seeded)');
  const cSeed = ui.number(secIn, { key: 'seed', label: 'seed', unit: '', value: 1 });
  const cIbase = ui.slider(secIn, { key: 'I_base', label: 'I_base', unit: 'pA', min: 0, max: 120, step: 1, value: 40.0 });
  const cIamp = ui.slider(secIn, { key: 'I_amp', label: 'I_amp', unit: 'pA', min: 0, max: 80, step: 1, value: 5.0 });

  const secH = ui.section(controls, 'Homeostasis');
  const cType = ui.select(secH, {
    key: 'type', label: 'type',
    options: [
      { value: 'activity', label: 'ActivityBaseHomeostasis' },
      { value: 'voltage', label: 'VoltageBaseHomeostasis' },
    ],
    value: 'activity',
  });
  // Activity-based. window_size is in steps; the target rate (Hz) is
  // activity_rate / (window_size * dt / 1000). Defaults: 1 spike / 1000 steps
  // (100 ms) at dt 0.1 ms = 10 Hz, a cortical rate.
  const cARate = ui.slider(secH, { key: 'activity_rate', label: 'activity_rate', unit: 'spikes/window', min: 1, max: 20, step: 1, value: 1 });
  const cWin = ui.slider(secH, { key: 'window_size', label: 'window_size', unit: 'steps', min: 100, max: 5000, step: 100, value: 1000 });
  const cUpd = ui.slider(secH, { key: 'updating_rate', label: 'updating_rate', unit: 'mV', min: 0, max: 1, step: 0.01, value: 0.2 });
  const cDecay = ui.slider(secH, { key: 'decay_rate', label: 'decay_rate', unit: '', min: 0.5, max: 1, step: 0.01, value: 1.0 });
  // Voltage-based
  const cTarget = ui.slider(secH, { key: 'target_voltage', label: 'target_voltage', unit: 'mV', min: -70, max: -40, step: 0.5, value: -60.0 });
  const cEta = ui.slider(secH, { key: 'eta_ip', label: 'eta_ip', unit: 'mV', min: 0, max: 0.01, step: 0.0005, value: 0.001 });

  const figRate = ui.plot(figures, { height: 220, xLabel: 't', yLabel: 'firing rate', xUnit: 'ms', yUnit: 'Hz' });
  const figCtrl = ui.plot(figures, { height: 200, xLabel: 't', yLabel: 'control variable', xUnit: 'ms', yUnit: 'mV' });
  const caption = document.createElement('p');
  caption.style.fontStyle = 'italic';
  figures.appendChild(caption);

  ui.note(notes, `Homeostasis keeps a population near a target. <b>ActivityBaseHomeostasis</b> counts spikes in
  a sliding window and shifts each neuron's threshold up or down so its activity tracks the target
  <var>activity_rate</var>. <b>VoltageBaseHomeostasis</b> instead pushes each neuron's voltage toward a target band
  by accumulating an "exhaustion" term. Both run on top of the same LIF dynamics under a random input.
  Units: v in mV, t in ms, I in pA, R in GOhm.`);
  ui.equation(notes, `activity: <var>threshold</var> ← <var>threshold</var> − <var>activities</var>·<var>updating_rate</var> (every <var>window_size</var> steps)<br>
  voltage: <var>exhaustion</var> += (<var>v</var>−band)·<var>eta_ip</var>, &nbsp; <var>v</var> ← <var>v</var> − <var>exhaustion</var>`);
  ui.note(notes, `The top figure shows the group firing rate (Hz) over 5 s; the dashed line is the target
  rate. The bottom shows the control variable (mean threshold for activity-based, mean voltage for
  voltage-based) converging toward the value that produces the target. The defaults aim for a cortical
  rate (~10 Hz): 1 spike per 100 ms window. The input (I_base 40 pA, I_amp 5 pA) is chosen so the
  starting rate is about 2× the target (~20 Hz); with R at the neuron default (0.5 GOhm) the threshold
  then settles within a couple of mV of the initial −50 mV, so it stays physiological. The caption
  reports the settled threshold. ML analogy: this is like a learning-rate / weight-decay loop that
  keeps activation levels in a healthy range; the analogy breaks because it is a slow, local,
  activity-driven rule with no error signal or label.`);
  ui.code(notes, () => codeSnippet());

  function params() {
    return {
      R: cR.get(), tau: cTau.get(), threshold: cTh.get(), v_reset: cReset.get(),
      v_rest: cRest.get(), dt: cDt.get(), I_base: cIbase.get(), I_amp: cIamp.get(),
    };
  }

  function run() {
    const p = params();
    const rng = ui.rng(cSeed.get());
    const type = cType.get();
    const vs = new Array(SIZE).fill(p.v_rest);
    const thresholds = new Array(SIZE).fill(p.threshold);
    const hos = type === 'activity'
      ? Array.from({ length: SIZE }, () => makeActivityHomeostasis({
          activity_rate: cARate.get(), window_size: cWin.get(),
          updating_rate: cUpd.get(), decay_rate: cDecay.get(),
        }))
      : Array.from({ length: SIZE }, () => makeVoltageHomeostasis({
          target_voltage: cTarget.get(), eta_ip: cEta.get(),
        }));

    const win = type === 'activity' ? cWin.get() : 1000;
    const targetHz = cARate.get() / (cWin.get() * p.dt / 1000);
    const t = [], rate = [], ctrl = [];
    let acc = 0;
    for (let it = 1; it <= STEPS; it++) {
      let groupSpikes = 0;
      for (let j = 0; j < SIZE; j++) {
        const I = p.I_base + rng.next() * p.I_amp;
        // The homeostasis threshold must drive the dynamics, not just be plotted.
        const pj = type === 'activity' ? { ...p, threshold: thresholds[j] } : p;
        const r = lifStep({ v: vs[j] }, pj, I, p.dt);
        vs[j] = r.v;
        groupSpikes += r.spike ? 1 : 0;
        if (type === 'activity') {
          thresholds[j] = hos[j].step(r.spike, thresholds[j], it);
        } else {
          vs[j] = hos[j].step(vs[j]);
        }
      }
      acc += groupSpikes;
      if (it % win === 0) {
        t.push(it * p.dt);
        rate.push(acc / (SIZE * win * p.dt / 1000)); // Hz
        ctrl.push(type === 'activity'
          ? thresholds.reduce((a, b) => a + b, 0) / SIZE
          : vs.reduce((a, b) => a + b, 0) / SIZE);
        acc = 0;
      }
    }
    figRate.draw([{ x: t, y: rate, label: 'firing rate', color: 'c1', style: 'line' }],
      { hlines: [{ y: targetHz, label: 'target' }], yRange: [0, Math.max(targetHz * 3, 50)] });
    if (type === 'activity') {
      figCtrl.draw([{ x: t, y: ctrl, label: 'mean threshold', color: 'c2', style: 'line' }], {});
      const finalTh = thresholds.reduce((a, b) => a + b, 0) / SIZE;
      caption.textContent = `settled mean threshold ${finalTh.toPrecision(4)} mV (start ${p.threshold} mV), ` +
        `final rate ${rate[rate.length - 1].toPrecision(3)} Hz (target ${targetHz.toPrecision(3)} Hz)`;
    } else {
      figCtrl.draw([{ x: t, y: ctrl, label: 'mean v', color: 'c2', style: 'line' }],
        { hlines: [{ y: cTarget.get(), label: 'target_voltage' }] });
      const finalV = vs.reduce((a, b) => a + b, 0) / SIZE;
      caption.textContent = `settled mean voltage ${finalV.toPrecision(4)} mV (target ${cTarget.get()} mV)`;
    }
  }

  function codeSnippet() {
    const p = params();
    return `from pymonntorch import Network, NeuronGroup
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons.specs import Fire
from conex.behaviors.neurons.homeostasis import ActivityBaseHomeostasis, VoltageBaseHomeostasis
from conex.behaviors.network.time_resolution import TimeResolution

net = Network(behavior=[TimeResolution(dt=${p.dt})])
pop = NeuronGroup(net=net, size=${SIZE}, behavior=[
    LIF(R=${p.R}, threshold=${cTh.get()}, tau=${p.tau},
        v_reset=${p.v_reset}, v_rest=${p.v_rest}),
    Fire(),
    ActivityBaseHomeostasis(activity_rate=${cARate.get()}, window_size=${cWin.get()},
                            updating_rate=${cUpd.get()}, decay_rate=${cDecay.get()}),
    # or: VoltageBaseHomeostasis(target_voltage=${cTarget.get()}, eta_ip=${cEta.get()}),
])
net.initialize()
pop.threshold = pop.vector(mode=${cTh.get()})
# set a random per-step input current (pA) on pop.I, then:
net.simulate_iterations(${STEPS})`;
  }

  const all = [cR, cTau, cTh, cReset, cRest, cDt, cSeed, cIbase, cIamp, cType,
    cARate, cWin, cUpd, cDecay, cTarget, cEta];
  for (const c of all) c.onInput(run);
  ui.onTheme(run);
  run();

  return { destroy() {} };
}
