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

  const sizeCtl = ui.number(controls, { key: 'size', label: 'SDR size n', unit: 'bits', value: 64, onInput: () => redraw() });
  const wCtl = ui.number(controls, { key: 'w', label: 'active bits w', unit: 'bits', value: 8, onInput: () => redraw() });
  const thetaCtl = ui.number(controls, { key: 'theta', label: 'match threshold θ', unit: 'bits', value: 4, onInput: () => redraw() });
  const seedCtl = ui.number(controls, { key: 'seed', label: 'random seed', unit: '', value: 1234, onInput: () => redraw() });
  const samplesCtl = ui.number(controls, { key: 'samples', label: 'Monte-Carlo samples', unit: '', value: 20000, onInput: () => redraw() });

  const bitsCanvas = ui.canvas(figures, { height: 120 });
  const probPlot = ui.plot(figures, { height: 220, xLabel: 'overlap b (bits)', yLabel: 'P(B = b)', xUnit: 'bits', yUnit: 'a.u.' });

  let last = {};

  function redraw() {
    const n = sizeCtl.get();
    const w = Math.min(wCtl.get(), n);
    const theta = thetaCtl.get();
    const rng = ui.rng(seedCtl.get());
    const pick = () => {
      const idx = new Set();
      while (idx.size < w) idx.add(Math.floor(rng.next() * n));
      return new SDR(n, [...idx]);
    };
    const a = pick();
    const b = pick();
    const union = a.union(b);
    const inter = a.intersection(b);
    const overlap = a.overlap(b);

    // Bits figure.
    const { canvas, ctx, width } = bitsCanvas;
    const rowH = 18;
    const pad = 8;
    ctx.clearRect(0, 0, width, bitsCanvas.height);
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const muted = ui.color('muted');
    drawBits(ctx, a.toDense(), pad, pad, width - 2 * pad, rowH, c1, muted);
    drawBits(ctx, b.toDense(), pad, pad + rowH + 6, width - 2 * pad, rowH, c2, muted);
    drawBits(ctx, union.toDense(), pad, pad + 2 * (rowH + 6), width - 2 * pad, rowH, c3, muted);
    drawBits(ctx, inter.toDense(), pad, pad + 3 * (rowH + 6), width - 2 * pad, rowH, ui.color('ink'), muted);

    // Hypergeometric PMF P(B=b) and the false-match tail P(B>=theta).
    const pmf = [];
    for (let b = 0; b <= w; b++) {
      // P(B=b) = C(w,b) C(n-w, w-b) / C(n,w); compute in log-space.
      const t = logCombLocal(w, b) + logCombLocal(n - w, w - b) - logCombLocal(n, w);
      pmf.push({ x: b, y: Math.exp(t) });
    }
    const pMatch = falseMatchProbability(n, w, w, theta);
    const pExp = expectedOverlap(n, w);

    // Monte-Carlo estimate of P(B>=theta).
    const m = samplesCtl.get();
    let hits = 0;
    const aDense = a.toDense();
    const rng2 = ui.rng(seedCtl.get() + 7919);
    for (let s = 0; s < m; s++) {
      const idx = new Set();
      while (idx.size < w) idx.add(Math.floor(rng2.next() * n));
      let ov = 0;
      for (const i of idx) if (aDense[i]) ov++;
      if (ov >= theta) hits++;
    }
    const pMC = hits / m;

    probPlot.draw(
      [
        { x: pmf.map((p) => p.x), y: pmf.map((p) => p.y), label: 'P(B=b)', color: 'c1', style: 'bars' },
      ],
      {
        xRange: [0, w],
        hlines: [{ y: pMatch, label: `P(B≥θ)=${pMatch.toExponential(2)}` }],
        vlines: [{ x: theta, label: `θ=${theta}` }],
      }
    );

    last = { n, w, theta, overlap, unionN: union.n_active, interN: inter.n_active, pMatch, pMC, pExp };
    updateNotes();
  }

  function updateNotes() {
    const { n, w, theta, overlap, unionN, interN, pMatch, pMC, pExp } = last;
    ui.note(
      notes,
      `<p>An SDR is a set of <b>${w}</b> active bits out of <b>${n}</b> (sparsity ${(100 * w / n).toFixed(1)}%). ` +
        `Two random SDRs share <b>${overlap}</b> bits here; the expected overlap of two independent codes is ` +
        `<b>w²/n = ${pExp.toFixed(3)} bits</b>. The union has <b>${unionN}</b> bits, the intersection <b>${interN}</b>.</p>` +
        `<p>The false-match probability is the chance a random ${w}-bit code overlaps a fixed one in at least ` +
        `<b>θ = ${theta}</b> bits: the hypergeometric tail ` +
        `<b>P(B≥θ) = ${pMatch.toExponential(3)}</b> (exact). ` +
        `A seeded Monte-Carlo estimate over ${samplesCtl.get()} samples gives <b>${pMC.toExponential(3)}</b> ` +
        `(${pMatch === 0 ? 'both below the sampling floor' : 'compare the two'}). ` +
        `All quantities are dimensionless counts (a.u.).</p>`
    );
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
    ui.code(notes, () =>
      `from conex import SDR\n` +
      `a = SDR(size=${n}, indices=...)   # ${w} active bits\n` +
      `b = SDR(size=${n}, indices=...)\n` +
      `a.overlap(b)          # ${overlap} bits\n` +
      `a.union(b).n_active   # ${unionN} bits\n` +
      `a.intersection(b).n_active  # ${interN} bits\n` +
      `a.match_threshold(b, theta=${theta})`
    );
  }

  redraw();
  ui.onTheme(redraw);
  return { destroy() {} };
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
