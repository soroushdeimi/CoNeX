import { encodeScalar } from '../js/models/htm.js';

export const meta = {
  id: 'htm-encoder',
  title: 'Scalar SDR encoder',
  conex: ['SDREncoder'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sizeCtl = ui.number(controls, { key: 'size', label: 'SDR size', unit: 'bits', value: 64, onInput: () => recompute() });
  const nActiveCtl = ui.number(controls, { key: 'n_active', label: 'active bits', unit: 'bits', value: 8, onInput: () => recompute() });
  const minCtl = ui.number(controls, { key: 'min_val', label: 'min value', unit: 'a.u.', value: 0.0, onInput: () => recompute() });
  const maxCtl = ui.number(controls, { key: 'max_val', label: 'max value', unit: 'a.u.', value: 1.0, onInput: () => recompute() });
  const valueCtl = ui.slider(controls, { key: 'value', label: 'reference value', unit: 'a.u.', min: 0, max: 1, step: 0.01, value: 0.5, onInput: () => recompute() });
  const deltaCtl = ui.slider(controls, { key: 'delta', label: 'distance Δ', unit: 'a.u.', min: 0, max: 0.5, step: 0.01, value: 0.02, onInput: () => recompute() });

  const bitsCanvas = ui.canvas(figures, { height: 90 });
  // ui.canvas does not expose its <figure>, and site.css is shell-owned, so the
  // caption is styled inline to match the .ui-figure figcaption look.
  const bitsCap = document.createElement('div');
  bitsCap.style.cssText = 'font-style:italic;color:var(--muted);font-size:0.85rem;margin-top:0.3rem;';
  bitsCap.textContent = 'active bits (teal = reference block, coral = sweeping block)';
  figures.appendChild(bitsCap);
  const overlapPlot = ui.plot(figures, { height: 220, xLabel: 'input value', yLabel: 'overlap', xUnit: 'a.u.', yUnit: 'bits' });

  function opts() {
    return { size: sizeCtl.get(), n_active: Math.min(nActiveCtl.get(), sizeCtl.get()), min_val: minCtl.get(), max_val: maxCtl.get() };
  }

  let run = null; // precomputed overlap curve over the value range
  let lastT = 0; // current player time, in value units

  // The player sweeps the input value across [min_val, max_val].
  const player = typeof ui.player === 'function'
    ? ui.player(figures, { duration: 1, unit: 'a.u.', playSeconds: 5, loop: true, onFrame: frame })
    : null;

  function frame(t) {
    if (!run) return;
    lastT = t;
    const { o, ref, delta, base, xs, ys } = run;
    const v = o.min_val + t;
    const cur = encodeScalar(v, o);
    drawBitsFig(base, cur);
    overlapPlot.draw([{ x: xs, y: ys, label: 'overlap', color: 'c1', style: 'step' }], {
      xRange: [o.min_val, o.max_val],
      yRange: [0, o.n_active],
      revealX: v,
      vlines: [{ x: ref, label: 'reference' }, { x: ref + delta, label: `ref+Δ` }],
    });
  }

  // Two rows: the reference block (fixed) and the sweeping block.
  function drawBitsFig(base, cur) {
    const { ctx, width, height } = bitsCanvas;
    ctx.clearRect(0, 0, width, height);
    const pad = 8;
    const rowH = 18;
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const muted = ui.color('muted');
    const n = run.o.size;
    const gap = 2;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    const baseDense = base.toDense();
    const curDense = cur.toDense();
    for (let i = 0; i < n; i++) {
      const x = pad + i * (bw + gap);
      ctx.fillStyle = baseDense[i] ? c1 : muted;
      ctx.fillRect(x, pad, bw, rowH);
      ctx.fillStyle = curDense[i] ? c2 : muted;
      ctx.fillRect(x, pad + rowH + 6, bw, rowH);
    }
  }
  // Canvas backing store resets on layout/resize, so register the draw via
  // onResize (called after layout and on every resize) rather than once.
  if (typeof bitsCanvas.onResize === 'function') bitsCanvas.onResize(() => {
    if (!run) return;
    const v = run.o.min_val + lastT;
    drawBitsFig(run.base, encodeScalar(v, run.o));
  });

  function recompute() {
    const o = opts();
    const ref = valueCtl.get();
    const delta = deltaCtl.get();
    const base = encodeScalar(ref, o);
    // Precompute the overlap with the reference across the whole range;
    // the player only indexes into this.
    const steps = 100;
    const xs = [];
    const ys = [];
    for (let s = 0; s <= steps; s++) {
      const v = o.min_val + ((o.max_val - o.min_val) * s) / steps;
      xs.push(v);
      ys.push(base.overlap(encodeScalar(v, o)));
    }
    run = { o, ref, delta, base, xs, ys };
    const span = Math.max(0, o.max_val - o.min_val);
    if (player) {
      player.setDuration(span);
      // Keep the current position (clamped) so a parameter change does not restart the sweep.
      player.seek(Math.min(lastT, span));
    } else {
      frame(span); // no player available: show the end of the sweep
    }
    updateNotes();
  }

  // Dynamic notes (readout + code) live in their own box, cleared each update
  // so they do not accumulate as the student changes controls.
  let dynNotes = null;

  function updateNotes() {
    if (!dynNotes) return;
    dynNotes.innerHTML = '';
    const { o, ref, delta, base } = run;
    const overlap = base.overlap(encodeScalar(ref + delta, o));
    ui.note(
      dynNotes,
      `<p>The scalar encoder maps a value to a contiguous block of <b>${o.n_active}</b> active bits whose center tracks the ` +
        `normalized value, wrapping around at the end. The animation sweeps the input value across ` +
        `[<b>${o.min_val}</b>, <b>${o.max_val}</b>] a.u.; the top row is the reference block at <b>${ref.toFixed(2)}</b>, the bottom row ` +
        `the sweeping block. Reference and reference+Δ (Δ = ${delta.toFixed(2)} a.u.) share <b>${overlap}</b> bits.</p>` +
        `<p>Overlap falls as the two values drift apart: nearby values keep most of their bits, distant values share none. ` +
        `This is how a continuous quantity becomes a sparse code that HTM can compare by overlap. ` +
        `Values are clamped to [<b>${o.min_val}</b>, <b>${o.max_val}</b>]. All values are dimensionless (a.u.).</p>`
    );
    ui.code(dynNotes, () =>
      `import torch\n` +
      `from pymonntorch import Network, NeuronGroup\n` +
      `from conex import SDREncoder, TimeResolution\n` +
      `net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")\n` +
      `neurons = NeuronGroup(net=net, size=${o.size},\n` +
      `    behavior={100: SDREncoder(size=${o.size}, n_active=${o.n_active}, encoder_type="scalar",\n` +
      `                              min_val=${o.min_val}, max_val=${o.max_val})}, tag="enc")\n` +
      `net.initialize()\n` +
      `enc = neurons.sdr_encoder\n` +
      `a = enc.encode_scalar(${ref.toFixed(2)})\n` +
      `b = enc.encode_scalar(${(ref + delta).toFixed(2)})\n` +
      `a.overlap(b)   # ${overlap} bits`
    );
  }

  function staticNotes() {
    ui.equation(
      notes,
      `<div>center = ⌊ <var>norm</var>(<var>v</var>) · (<var>n</var> − <var>w</var>) ⌋, &nbsp; norm(v) = clamp((v − min)/(max − min), 0, 1)</div>`
    );
    ui.note(
      notes,
      `<p><b>Background.</b> Real sensory input is continuous; the encoder turns it into an SDR the Spatial Pooler can use. ` +
        `A sliding-window (bucket) encoder keeps nearby values similar. ` +
        `<b>ML analogy:</b> a quantizer / one-hot with a soft neighborhood. ` +
        `The analogy breaks because the wrap-around makes the space circular, so the two ends of the range are neighbors.</p>`
    );
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}
