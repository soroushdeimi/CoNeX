import { SDR, SDRClassifier } from '../js/models/htm.js';

export const meta = {
  id: 'htm-classifier',
  title: 'SDR classifier',
  conex: ['SDRClassifier'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const nClassesCtl = ui.number(controls, { key: 'n_classes', label: 'n_classes', unit: 'classes', value: 3, onInput: () => recompute() });
  const sdrSizeCtl = ui.number(controls, { key: 'sdr_size', label: 'sdr_size', unit: 'bits', value: 64, onInput: () => recompute() });
  const lrCtl = ui.slider(controls, { key: 'learning_rate', label: 'learning_rate', unit: 'a.u.', min: 0.001, max: 0.1, step: 0.001, value: 0.01, onInput: () => recompute() });
  const examplesCtl = ui.number(controls, { key: 'examples', label: 'examples per class', unit: '', value: 5, onInput: () => recompute() });

  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => recompute() });

  const scoresPlot = ui.plot(figures, { height: 220, xLabel: 'class', yLabel: 'score', xUnit: 'index', yUnit: 'a.u.' });
  const bitsCanvas = ui.canvas(figures, { height: 90 });
  const readout = document.createElement('div');
  readout.className = 'ui-note';
  figures.appendChild(readout);

  let run = null; // precomputed training + inference run
  let lastT = 0; // current player time

  // The player plays training examples one by one, then the inference.
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'events', playSeconds: 5, loop: true, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const T = run.total;
    const tt = Math.max(0, Math.min(T, t));
    let bars; // one value per class
    let label;
    if (tt < run.nTrain) {
      // Training: e examples applied; bars show the per-class weight sums.
      const e = Math.min(run.nTrain - 1, Math.floor(tt));
      const w = run.weightsAfter[e];
      bars = run.classIdx.map((c) => {
        let s = 0;
        for (let i = 0; i < run.n; i++) s += w[c][i];
        return s;
      });
      const ex = run.trainSeq[e];
      label = `training: example ${e + 1}/${run.nTrain} (class ${ex.c})`;
      drawBitsFig(ex.dense);
    } else {
      // Inference: reveal the test SDR bit by bit; scores accumulate.
      const k = Math.min(run.n, Math.floor(tt - run.nTrain));
      const w = run.weightsAfter[run.nTrain - 1];
      bars = run.classIdx.map((c) => {
        let s = 0;
        for (let i = 0; i < k; i++) s += w[c][i] * run.testDense[i];
        return s / (run.counts[c] + 1e-6);
      });
      label = `inference: test SDR, ${k}/${run.n} bits revealed`;
      drawBitsFig(run.testDense.slice(0, k).concat(new Array(run.n - k).fill(0)));
    }
    const maxBar = Math.max(...bars, 1e-6);
    scoresPlot.draw([{ x: run.classIdx, y: bars, label: tt < run.nTrain ? 'weight sum' : 'score', color: 'c1', style: 'bars' }], {
      xRange: [0, run.classIdx.length],
      yRange: [0, maxBar * 1.1],
    });
    readout.textContent = label;
  }

  // One row of bits: the current training pattern or the revealed test SDR.
  function drawBitsFig(dense) {
    const { ctx, width, height } = bitsCanvas;
    ctx.clearRect(0, 0, width, height);
    const n = run.n;
    const pad = 8;
    const gap = 2;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = dense[i] ? ui.color('c1') : ui.color('muted');
      ctx.fillRect(pad + i * (bw + gap), pad, bw, 18);
    }
  }
  // The canvas backing store is reset on layout/resize, so register the draw
  // via onResize (called after layout and on every resize) rather than once.
  if (typeof bitsCanvas.onResize === 'function') bitsCanvas.onResize(() => frame(lastT));

  function recompute() {
    const n = sdrSizeCtl.get();
    const k = nClassesCtl.get();
    const ex = examplesCtl.get();
    const block = Math.max(1, Math.floor(n / k));
    const patterns = [];
    for (let c = 0; c < k; c++) {
      const start = c * block;
      const idx = [];
      for (let i = 0; i < block; i++) idx.push((start + i) % n);
      patterns.push(new SDR(n, idx));
    }
    const clf = new SDRClassifier({ n_classes: k, sdr_size: n, learning_rate: lrCtl.get() });
    // Training order: round-robin over classes, example by example.
    const trainSeq = [];
    for (let i = 0; i < ex; i++) for (let c = 0; c < k; c++) trainSeq.push({ c, dense: patterns[c].toDense() });
    // Apply the examples up front, snapshotting the weights after each one;
    // the player only indexes into these snapshots.
    const weightsAfter = [];
    for (const e of trainSeq) {
      clf.learn(new SDR(n, e.dense.map((v, i) => (v ? i : -1)).filter((i) => i >= 0)), e.c);
      weightsAfter.push(clf.weights.map((w) => w.slice()));
    }
    const testDense = patterns[0].toDense();
    const [pred, scores] = clf.infer(patterns[0]);
    run = {
      n, classIdx: Array.from({length: k}, (_, c) => c), ex, patterns, trainSeq, weightsAfter,
      nTrain: trainSeq.length,
      counts: Array.from(clf.counts),
      testDense, pred, scores,
      total: trainSeq.length + n,
    };
    if (player) {
      player.setDuration(run.total);
      // Keep the current position (clamped) so a parameter change does not restart the run.
      player.seek(Math.min(lastT, run.total));
    } else {
      frame(run.total); // no player available: show the finished run
    }
    updateNotes();
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student changes controls.
  let dynNotes = null;

  function updateNotes() {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const k = run.classIdx.length;
    const { n, ex, pred, scores } = run;
    ui.note(
      dynNotes,
      `<p>The SDR classifier stores, for each class, a weight vector over the SDR bits. <b>learn()</b> adds ` +
        `<b>learning_rate = ${lrCtl.get()}</b> a.u. to the active bits of the labelled class; <b>infer()</b> scores each class as ` +
        `(weights · dense) / (count + 1e-6) and returns the argmax. The animation plays the ${run.nTrain} training examples ` +
        `(${ex} per class) one by one — the bars show the per-class weight sums growing — then runs inference on the class-0 code, ` +
        `revealing its bits one at a time so the scores accumulate. Final scores: ` +
        `<b>[${scores.map((s) => s.toFixed(4)).join(', ')}]</b> a.u.; the test code is classified as <b>class ${pred}</b>.</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from pymonntorch import Network, NeuronGroup\n` +
      `from conex import SDRClassifier, SDR, TimeResolution\n` +
      `net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")\n` +
      `neurons = NeuronGroup(net=net, size=${n},\n` +
      `    behavior={100: SDRClassifier(n_classes=${k}, sdr_size=${n},\n` +
      `                                  learning_rate=${lrCtl.get()})}, tag="clf")\n` +
      `net.initialize()\n` +
      `clf = neurons.sdr_classifier\n` +
      `patterns = [\n` +
      `    (0, SDR(size=${n}, indices=torch.arange(0, 8))),\n` +
      `    (1, SDR(size=${n}, indices=torch.arange(20, 28))),\n` +
      `    (2, SDR(size=${n}, indices=torch.arange(40, 48))),\n` +
      `]\n` +
      `for label, sdr in patterns:\n` +
      `    for _ in range(${ex}):\n` +
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
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}
