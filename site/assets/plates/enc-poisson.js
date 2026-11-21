// Plate: Poisson & latency encoding of a 16x16 grayscale pattern.
import * as M from '../js/models/plasticity.js';

export const meta = {
  id: 'enc-poisson',
  title: 'Poisson & latency encoding',
  conex: ['SimplePoisson', 'Poisson', 'Intensity2Latency'],
  source: 'conex/helpers/transforms/encoders.py',
};

// A digit-like "7" stroke on a 16x16 grid, values in [0,1]. Procedural, no data.
function pattern16() {
  const img = new Array(256).fill(0);
  for (let x = 0; x < 16; x++) { img[2 * 16 + x] = 1.0; img[3 * 16 + x] = 0.6; }
  for (let y = 2; y < 14; y++) {
    const x = Math.floor(13 - (y - 2) * 0.8);
    img[y * 16 + x] = 1.0;
    if (x + 1 < 16) img[y * 16 + x + 1] = 0.6;
  }
  return img;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);
  const img = pattern16();
  const N = 256;

  const raster = ui.raster(figures, { height: 220, xLabel: 'time', xUnit: 'steps' });
  const heat = ui.heatmap(figures, { rows: 16, cols: 16, label: 'spikes per pixel' });

  function encode() {
    const T = state.time_window;
    const r = state.ratio;
    const s = state.seed;
    let spikes;
    if (state.encoder === 'i2l') spikes = M.intensity2Latency(img, { time_window: T });
    else if (state.encoder === 'poisson') spikes = M.poisson(img, { time_window: T, ratio: r, seed: s });
    else spikes = M.simplePoisson(img, { time_window: T, ratio: r, seed: s });
    const ticks = [];
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) if (spikes[t * N + i]) ticks.push([t, i]);
    raster.draw(ticks, N, T);
    const counts = new Array(N).fill(0);
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) if (spikes[t * N + i]) counts[i]++;
    heat.draw(counts, { min: 0, max: Math.max(1, ...counts) });
  }

  const state = { time_window: 30, ratio: 5 / 30, seed: 42, encoder: 'simple' };
  const sec = ui.section(controls, 'Encoder');
  ui.slider(sec, { key: 'time_window', label: 'time_window', unit: 'steps', min: 10, max: 100, step: 1, value: 30, onInput: (v) => { state.time_window = v; encode(); } });
  ui.slider(sec, { key: 'ratio', label: 'ratio', unit: 'a.u.', min: 0.05, max: 1.0, step: 0.05, value: 5 / 30, onInput: (v) => { state.ratio = v; encode(); } });
  ui.number(sec, { key: 'seed', label: 'seed', unit: '', value: 42, onInput: (v) => { state.seed = v; encode(); } });
  ui.select(sec, {
    key: 'encoder', label: 'encoder', value: 'simple',
    options: [
      { value: 'simple', label: 'SimplePoisson' },
      { value: 'poisson', label: 'Poisson' },
      { value: 'i2l', label: 'Intensity2Latency' },
    ],
    onInput: (v) => { state.encoder = v; encode(); },
  });

  ui.onTheme(encode);
  encode();

  ui.note(notes, `A grayscale image (values 0-1) is turned into a spike train. <b>SimplePoisson</b> and <b>Poisson</b> are rate codes: a pixel with intensity <var>i</var> spikes at rate <var>i</var>·ratio, so brighter pixels fire more often. <b>Intensity2Latency</b> is a latency code: brighter pixels spike <i>sooner</i>, and the first spike carries the value. In the retina, ganglion cells encode light intensity both by firing rate and by response latency.`);
  ui.equation(notes, `<var>spike</var>(<var>t</var>) ~ Bernoulli(<var>i</var> · ratio) &nbsp;for SimplePoisson; &nbsp; intervals ~ Poisson(1 / (<var>i</var> · ratio)) &nbsp;for Poisson.`);
  ui.note(notes, `The ratio is dimensionless (a.u.); time is in simulation steps. ML analogy: this is like turning a continuous feature vector into a stochastic binary code (e.g. a dropout-style or stochastic binary embedding). The analogy breaks because real neurons have refractory periods and adaptation that a per-step Bernoulli does not model.`);
  ui.code(notes, () => {
    const cls = state.encoder === 'i2l' ? 'Intensity2Latency' : state.encoder === 'poisson' ? 'Poisson' : 'SimplePoisson';
    const ratioArg = state.encoder === 'i2l' ? '' : `, ratio=${state.ratio}`;
    return `from conex.helpers.transforms.encoders import ${cls}
img = ...  # 16x16, values in [0, 1]
spikes = ${cls}(time_window=${state.time_window}${ratioArg})(img)`;
  });
  return { destroy() {} };
}
