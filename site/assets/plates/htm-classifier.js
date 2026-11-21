import { SDR, SDRClassifier } from '../js/models/htm.js';

export const meta = {
  id: 'htm-classifier',
  title: 'SDR classifier',
  conex: ['SDRClassifier'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const nClassesCtl = ui.number(controls, { key: 'n_classes', label: 'n_classes', unit: 'classes', value: 3, onInput: () => reset() });
  const sdrSizeCtl = ui.number(controls, { key: 'sdr_size', label: 'sdr_size', unit: 'bits', value: 64, onInput: () => reset() });
  const lrCtl = ui.slider(controls, { key: 'learning_rate', label: 'learning_rate', unit: 'a.u.', min: 0.001, max: 0.1, step: 0.001, value: 0.01, onInput: () => reset() });
  const examplesCtl = ui.number(controls, { key: 'examples', label: 'examples per class', unit: '', value: 5, onInput: () => reset() });

  const trainBtn = ui.button(controls, { label: 'Train', onClick: () => train() });
  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => reset() });

  const scoresPlot = ui.plot(figures, { height: 220, xLabel: 'class', yLabel: 'score', xUnit: 'index', yUnit: 'a.u.' });
  const bitsCanvas = ui.canvas(figures, { height: 90 });

  let clf = null;
  let patterns = []; // one SDR per class
  let lastResult = null;

  function buildPatterns() {
    const n = sdrSizeCtl.get();
    const k = nClassesCtl.get();
    const block = Math.max(1, Math.floor(n / k));
    patterns = [];
    for (let c = 0; c < k; c++) {
      const start = c * block;
      const idx = [];
      for (let i = 0; i < block; i++) idx.push((start + i) % n);
      patterns.push(new SDR(n, idx));
    }
    return patterns;
  }

  function train() {
    if (!clf) return;
    const ex = examplesCtl.get();
    for (let c = 0; c < patterns.length; c++) {
      for (let i = 0; i < ex; i++) clf.learn(patterns[c], c);
    }
    const [pred, scores] = clf.infer(patterns[0]);
    lastResult = { pred, scores };
    redraw();
  }

  function reset() {
    clf = new SDRClassifier({ n_classes: nClassesCtl.get(), sdr_size: sdrSizeCtl.get(), learning_rate: lrCtl.get() });
    patterns = buildPatterns();
    lastResult = null;
    train(); // train on load so the figure and note show real content
  }

  // Canvas backing store resets on layout/resize, so register the draw via
  // onResize (called after layout and on every resize) rather than once.
  function drawBitsFig() {
    if (!patterns.length) return;
    const { ctx, width, height } = bitsCanvas;
    ctx.clearRect(0, 0, width, height);
    const n = sdrSizeCtl.get();
    const dense = patterns[0].toDense();
    const pad = 8;
    const gap = 2;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = dense[i] ? ui.color('c1') : ui.color('muted');
      ctx.fillRect(pad + i * (bw + gap), pad, bw, 18);
    }
  }
  // Register the redraw on resize when the ui supports it (the real ui.js does);
  // the direct drawBitsFig() call in redraw() covers the no-onResize case.
  if (typeof bitsCanvas.onResize === 'function') bitsCanvas.onResize(drawBitsFig);

  function redraw() {
    if (!clf) return;
    drawBitsFig();

    // Output distribution: scores per class. Bars start at 0.
    const scores = lastResult ? lastResult.scores : new Array(nClassesCtl.get()).fill(0);
    const xs = scores.map((_, c) => c);
    const maxScore = Math.max(...scores, 1e-6);
    scoresPlot.draw([{ x: xs, y: scores, label: 'score', color: 'c1', style: 'bars' }], {
      xRange: [0, nClassesCtl.get()],
      yRange: [0, maxScore * 1.1],
    });

    updateNotes();
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student changes controls.
  let dynNotes = null;

  function updateNotes() {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const k = nClassesCtl.get();
    const pred = lastResult ? lastResult.pred : null;
    ui.note(
      dynNotes,
      `<p>The SDR classifier stores, for each class, a weight vector over the SDR bits. <b>learn()</b> adds ` +
        `<b>learning_rate = ${lrCtl.get()}</b> to the active bits of the labelled class; <b>infer()</b> scores each class as ` +
        `(weights · dense) / (count + 1e-6) and returns the argmax. After training on ${examplesCtl.get()} examples per class, ` +
        `the first code is classified as <b>${pred === null ? '—' : 'class ' + pred}</b>.</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from pymonntorch import Network, NeuronGroup\n` +
      `from conex import SDRClassifier, SDR, TimeResolution\n` +
      `net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")\n` +
      `neurons = NeuronGroup(net=net, size=${sdrSizeCtl.get()},\n` +
      `    behavior={100: SDRClassifier(n_classes=${k}, sdr_size=${sdrSizeCtl.get()},\n` +
      `                                  learning_rate=${lrCtl.get()})}, tag="clf")\n` +
      `net.initialize()\n` +
      `clf = neurons.sdr_classifier\n` +
      `patterns = [\n` +
      `    (0, SDR(size=${sdrSizeCtl.get()}, indices=torch.arange(0, 8))),\n` +
      `    (1, SDR(size=${sdrSizeCtl.get()}, indices=torch.arange(20, 28))),\n` +
      `    (2, SDR(size=${sdrSizeCtl.get()}, indices=torch.arange(40, 48))),\n` +
      `]\n` +
      `for label, sdr in patterns:\n` +
      `    for _ in range(${examplesCtl.get()}):\n` +
      `        clf.learn(neurons, sdr, label)\n` +
      `pred, scores = clf.infer(neurons, patterns[0][1])`
    );
  }

  function staticNotes() {
    ui.note(
      notes,
      `<p><b>Background.</b> This is a minimal content-addressable readout: the SDR is the input, the class weights are the stored ` +
        `associations. <b>ML analogy:</b> a linear probe / naive Bayes over sparse features. The analogy breaks because there is no ` +
        `bias term, no softmax, and the score is normalized by how often each class was seen, so a rarely-trained class is penalized.</p>`
    );
    ui.equation(
      notes,
      `<div>score<sub>c</sub> = (<var>w</var><sub>c</sub> · <var>x</var>) / (count<sub>c</sub> + 1e-6), &nbsp; ` +
        `predict = arg<sub>c</sub> score<sub>c</sub></div>`
    );
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  reset();
  ui.onTheme(redraw);
  return { destroy() {} };
}
