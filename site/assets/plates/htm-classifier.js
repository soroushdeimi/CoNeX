import { SDR, SDRClassifier } from '../js/models/htm.js';
import { ease } from '../js/core.js';

export const meta = {
  id: 'htm-classifier',
  title: 'SDR classifier',
  conex: ['SDRClassifier'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

// Pure hero state for the SDR classifier plate. During training the current
// example "flies" into its class bucket (fly = eased progress 0..1) and the
// per-class weight-sum bars grow; during inference the test SDR pulses and the
// score bars race. t is the player time, in events.
export function clfHeroState(run, t) {
  const T = run.total;
  const tt = Math.max(0, Math.min(T, t));
  const n = run.n;
  if (tt < run.nTrain) {
    const e = Math.min(run.nTrain - 1, Math.floor(tt));
    const f = ease(tt - e);
    const ex = run.trainSeq[e];
    const w = run.weightsAfter[e];
    const bars = run.classIdx.map((c) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += w[c][i];
      return s;
    });
    return {
      phase: 'train', e, f, fly: f, class: ex.c, dense: ex.dense, bars,
      narration: `training: example ${e + 1}/${run.nTrain} (class ${ex.c}) flies into its bucket; weight bars grow`,
    };
  }
  const kbits = Math.min(n, Math.floor(tt - run.nTrain));
  const f = ease(tt - run.nTrain - kbits);
  const w = run.weightsAfter[run.nTrain - 1];
  const bars = run.classIdx.map((c) => {
    let s = 0;
    for (let i = 0; i < kbits; i++) s += w[c][i] * run.testDense[i];
    return s / (run.counts[c] + 1e-6);
  });
  return {
    phase: 'infer', kbits, f, fly: 1, class: run.pred, dense: run.testDense, bars,
    narration: `inference: test SDR pulses, ${kbits}/${n} bits revealed; score bars race`,
  };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const nClassesCtl = ui.number(controls, { key: 'n_classes', label: 'n_classes', unit: 'classes', value: 3, onInput: () => recompute() });
  const sdrSizeCtl = ui.number(controls, { key: 'sdr_size', label: 'sdr_size', unit: 'bits', value: 64, onInput: () => recompute() });
  const lrCtl = ui.slider(controls, { key: 'learning_rate', label: 'learning_rate', unit: 'a.u.', min: 0.001, max: 0.1, step: 0.001, value: 0.01, onInput: () => recompute() });
  const examplesCtl = ui.number(controls, { key: 'examples', label: 'examples per class', unit: '', value: 5, onInput: () => recompute() });

  const resetBtn = ui.button(controls, { label: 'Reset', onClick: () => recompute() });

  // Hero: the SDR flies into its class bucket during training; during
  // inference the test SDR pulses and the score bars race.
  const hero = ui.hero(figures, { height: 200 });
  const heroCap = { set(text) { hero.narration(text); } };
  const scoresPlot = ui.plot(figures, { height: 220, xLabel: 'class', yLabel: 'score', xUnit: 'index', yUnit: 'a.u.' });
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
    const st = clfHeroState(run, lastT);
    drawHero(st);
    const maxBar = Math.max(...st.bars, 1e-6);
    scoresPlot.draw([{ x: run.classIdx, y: st.bars, label: st.phase === 'train' ? 'weight sum' : 'score', color: 'c1', style: 'bars' }], {
      xRange: [0, run.classIdx.length],
      yRange: [0, maxBar * 1.1],
    });
    readout.textContent = st.narration;
  }

  // Hero: a row of class buckets. The current SDR (teal bits) flies into its
  // bucket during training (fly = eased 0..1); during inference the test SDR
  // pulses (fly = 1) and the winning bucket is highlighted.
  function drawHero(st) {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const n = run.n;
    const k = run.classIdx.length;
    const pad = 8;
    const bucketW = (width - 2 * pad) / k;
    const sdrH = 22;
    const sdrY = pad;
    const bucketY = pad + sdrH + 24;
    const bucketH = height - bucketY - pad;
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const muted = ui.color('muted');
    const ink = ui.color('ink');

    // Class buckets.
    for (let c = 0; c < k; c++) {
      const x = pad + c * bucketW;
      const isTarget = c === st.class;
      ctx.strokeStyle = isTarget ? c2 : muted;
      ctx.lineWidth = isTarget ? 2 : 1;
      ctx.strokeRect(x + 2, bucketY, bucketW - 4, bucketH);
      ctx.fillStyle = ink;
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`class ${c}`, x + bucketW / 2, bucketY + bucketH - 16);
    }

    // The SDR: a row of bits that flies from the top into its target bucket.
    const gap = 2;
    const bw = (bucketW - 8 - gap * (n - 1)) / n;
    const targetX = pad + st.class * bucketW + 4;
    const startX = pad + (width - 2 * pad) / 2 - (n * (bw + gap) - gap) / 2;
    const x = startX + (targetX - startX) * st.fly;
    const y = sdrY + (bucketY - sdrH - 8 - sdrY) * st.fly;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = st.dense[i] ? c1 : muted;
      ctx.fillRect(x + i * (bw + gap), y, bw, sdrH);
    }
    heroCap.set(st.narration);
  }
  hero.onResize(() => { if (run) drawHero(clfHeroState(run, lastT)); });

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
        `bias term, no softmax, and the score is divided by how often each class was seen, giving the ` +
        `average weight per presentation, so a frequently-trained class does not win just by count.</p>`
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
