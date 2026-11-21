import { SDR, falseMatchProbability, expectedOverlap } from '../js/models/htm.js';

export const meta = {
  id: 'htm-sdr',
  title: 'Sparse Distributed Representations',
  conex: ['SDR', 'SDRConfig', 'UnionSDR'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

// Draw a row of bits: active bits filled, inactive hollow.
function drawBits(ctx, bits, x0, y, w, h, color, muted) {
  const n = bits.length;
  const gap = 2;
  const bw = (w - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const x = x0 + i * (bw + gap);
    ctx.fillStyle = bits[i] ? color : muted;
    ctx.fillRect(x, y, bw, h);
  }
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sizeCtl = ui.number(controls, { key: 'size', label: 'SDR size n', unit: 'bits', value: 64, onInput: () => recompute() });
  const wCtl = ui.number(controls, { key: 'w', label: 'active bits w', unit: 'bits', value: 8, onInput: () => recompute() });
  const thetaCtl = ui.number(controls, { key: 'theta', label: 'match threshold θ', unit: 'bits', value: 4, onInput: () => recompute() });
  const seedCtl = ui.number(controls, { key: 'seed', label: 'random seed', unit: '', value: 1234, onInput: () => recompute() });
  const samplesCtl = ui.number(controls, { key: 'samples', label: 'Monte-Carlo samples', unit: '', value: 20000, onInput: () => recompute() });

  const bitsCanvas = ui.canvas(figures, { height: 120 });
  // ui.canvas does not expose its <figure>, and site.css is shell-owned, so the
  // caption is styled inline to match the .ui-figure figcaption look.
  const bitsCap = document.createElement('div');
  bitsCap.style.cssText = 'font-style:italic;color:var(--muted);font-size:0.85rem;margin-top:0.3rem;';
  bitsCap.textContent = 'bits (teal = fixed code a, coral = current random sample, ink = their intersection)';
  figures.appendChild(bitsCap);
  const histPlot = ui.plot(figures, { height: 220, xLabel: 'overlap b', yLabel: 'P(B = b)', xUnit: 'bits', yUnit: 'a.u.' });
  const readout = document.createElement('div');
  readout.className = 'ui-note';
  figures.appendChild(readout);

  let run = null; // precomputed sampling run
  let lastT = 0; // current player time, in samples

  // The player reveals the precomputed run sample by sample.
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'samples', playSeconds: 5, loop: true, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const m = run.m;
    const idx = Math.max(0, Math.min(m - 1, Math.floor(t)));
    drawBitsFig(idx);

    // Histogram of the overlaps so far, normalized to a probability,
    // against the exact hypergeometric PMF.
    const counts = run.cum[idx];
    const hist = counts.map((c, b) => ({ x: b, y: c / (idx + 1) }));
    histPlot.draw(
      [
        { x: hist.map((p) => p.x), y: hist.map((p) => p.y), label: 'Monte-Carlo', color: 'c2', style: 'bars' },
        { x: run.pmf.map((p) => p.x), y: run.pmf.map((p) => p.y), label: 'P(B=b) exact', color: 'c1', style: 'line' },
      ],
      {
        xRange: [0, run.w],
        yRange: [0, run.maxPmf * 1.1],
        // The hline label would sit on the bars at the low pMatch level, so the
        // value is shown in the top vline label instead; the hline stays as a guide.
        hlines: [{ y: run.pMatch, label: '' }],
        vlines: [{ x: run.theta, label: `θ=${run.theta} · P(B≥θ)=${run.pMatch.toExponential(2)}` }],
      }
    );
    readout.textContent =
      `sample ${idx + 1}/${m}: overlap ${run.ov[idx]} bits, ` +
      `P̂(B≥θ) = ${(run.hits[idx] / (idx + 1)).toExponential(3)}`;
  }

  // Rows: fixed code a, the current random sample, their intersection.
  function drawBitsFig(idx) {
    const { ctx, width, height } = bitsCanvas;
    const rowH = 18;
    const pad = 8;
    ctx.clearRect(0, 0, width, height);
    const a = run.a.toDense();
    const s = run.bits[idx];
    const inter = a.map((v, i) => (v && s[i] ? 1 : 0));
    drawBits(ctx, a, pad, pad, width - 2 * pad, rowH, ui.color('c1'), ui.color('muted'));
    drawBits(ctx, s, pad, pad + rowH + 6, width - 2 * pad, rowH, ui.color('c2'), ui.color('muted'));
    drawBits(ctx, inter, pad, pad + 2 * (rowH + 6), width - 2 * pad, rowH, ui.color('ink'), ui.color('muted'));
  }
  // The canvas backing store is reset on layout/resize, so register the draw
  // via onResize (called after layout and on every resize) rather than once.
  if (typeof bitsCanvas.onResize === 'function') bitsCanvas.onResize(() => drawBitsFig(Math.max(0, Math.min(run ? run.m - 1 : 0, Math.floor(lastT)))));

  function recompute() {
    const n = sizeCtl.get();
    const w = Math.min(wCtl.get(), n);
    const theta = thetaCtl.get();
    const m = Math.max(1, samplesCtl.get());
    const rng = ui.rng(seedCtl.get());
    const pick = () => {
      const idx = new Set();
      while (idx.size < w) idx.add(Math.floor(rng.next() * n));
      return new SDR(n, [...idx]);
    };
    const a = pick();
    const aDense = a.toDense();
    // Draw all m random codes up front; the player only indexes into this.
    const rng2 = ui.rng(seedCtl.get() + 7919);
    const bits = new Array(m);
    const ov = new Array(m);
    const hits = new Array(m);
    const cum = new Array(m);
    const counts = new Array(w + 1).fill(0);
    let h = 0;
    for (let s = 0; s < m; s++) {
      const idx = new Set();
      while (idx.size < w) idx.add(Math.floor(rng2.next() * n));
      const d = new Array(n).fill(0);
      let o = 0;
      for (const i of idx) {
        d[i] = 1;
        if (aDense[i]) o++;
      }
      bits[s] = d;
      ov[s] = o;
      counts[o]++;
      if (o >= theta) h++;
      hits[s] = h;
      cum[s] = counts.slice();
    }
    // Hypergeometric PMF P(B=b), computed in log-space.
    const pmf = [];
    for (let b = 0; b <= w; b++) {
      const t = logCombLocal(w, b) + logCombLocal(n - w, w - b) - logCombLocal(n, w);
      pmf.push({ x: b, y: Math.exp(t) });
    }
    const maxPmf = Math.max(...pmf.map((p) => p.y), 1e-12);
    run = {
      n, w, theta, m, a, bits, ov, hits, cum, pmf, maxPmf,
      b0: new SDR(n, bits[0].map((v, i) => (v ? i : -1)).filter((i) => i >= 0)),
      pMatch: falseMatchProbability(n, w, w, theta),
      pExp: expectedOverlap(n, w),
    };
    if (player) {
      player.setDuration(m);
      // Keep the current position (clamped) so a parameter change does not restart the run.
      player.seek(Math.min(lastT, m));
    } else {
      frame(m); // no player available: show the finished run
    }
    updateNotes();
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student changes controls.
  let dynNotes = null;

  function updateNotes() {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const { n, w, theta, m, a, b0, ov, hits, pMatch, pExp } = run;
    const pMC = hits[m - 1] / m;
    ui.note(
      dynNotes,
      `<p>An SDR is a set of <b>${w}</b> active bits out of <b>${n}</b> (sparsity ${(100 * w / n).toFixed(1)}%). ` +
        `The fixed code <b>a</b> (top row) is compared against <b>${m}</b> random ${w}-bit codes drawn one at a time ` +
        `(middle row); the bottom row is their intersection. The expected overlap of two independent codes is ` +
        `<b>w²/n = ${pExp.toFixed(3)} bits</b>.</p>` +
        `<p>The false-match probability is the chance a random ${w}-bit code overlaps a fixed one in at least ` +
        `<b>θ = ${theta}</b> bits: the hypergeometric tail ` +
        `<b>P(B≥θ) = ${pMatch.toExponential(3)}</b> (exact). ` +
        `The Monte-Carlo estimate after all ${m} samples is <b>${pMC.toExponential(3)}</b>. ` +
        `All quantities are dimensionless counts (a.u.).</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from conex import SDR\n` +
      `a = SDR(size=${n}, indices=torch.tensor([${a.indices.join(', ')}]))\n` +
      `b = SDR(size=${n}, indices=torch.tensor([${b0.indices.join(', ')}]))\n` +
      `a.overlap(b)                    # ${ov[0]} bits\n` +
      `a.union(b).n_active             # ${a.union(b0).n_active} bits\n` +
      `a.intersection(b).n_active      # ${ov[0]} bits\n` +
      `a.match_threshold(b, theta=${theta})`
    );
  }

  function staticNotes() {
    ui.equation(
      notes,
      `<div>P(B ≥ θ) = <var>Σ</var><sub>b=θ</sub><sup>w</sup> C(<var>w</var>, b)·C(<var>n</var>−<var>w</var>, <var>w</var>−b) / C(<var>n</var>, <var>w</var>)</div>`
    );
    ui.note(
      notes,
      `<p><b>Background.</b> SDRs are the data structure of HTM: high-dimensional, sparse, and compared by overlap. ` +
        `Partial matches still work (fault tolerance), and a union can hold several items at once. ` +
        `<b>ML analogy:</b> a sparse binary hash with a similarity threshold; the union behaves like a Bloom filter. ` +
        `The analogy breaks because cortical codes are structured and plastic, while the formula assumes independent uniform codes.</p>` +
      `<p><b>Source of the false-match formula.</b> It is the hypergeometric overlap tail from ` +
        `S. Ahmad and J. Hawkins, <i>"How do neurons operate on sparse distributed representations? A mathematical theory of ` +
        `sparsity, neurons and active dendrites"</i>, arXiv:1601.00720 (2016), eq. 3. This formula is <b>not</b> part of the CoNeX ` +
        `<code>SDR</code> class — CoNeX's <code>SDR</code> provides overlap/union/intersection but not this probability; it is shown here ` +
        `as the standard HTM literature result.</p>`
    );
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}

// Local log-comb for the PMF bars (same Lanczos lgamma as the model).
function lgamma(x) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function logCombLocal(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}
