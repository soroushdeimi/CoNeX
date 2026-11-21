import { encodeScalar } from '../js/models/htm.js';

export const meta = {
  id: 'htm-encoder',
  title: 'Scalar SDR encoder',
  conex: ['SDREncoder'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sizeCtl = ui.number(controls, { key: 'size', label: 'SDR size', unit: 'bits', value: 64, onInput: () => redraw() });
  const nActiveCtl = ui.number(controls, { key: 'n_active', label: 'active bits', unit: 'bits', value: 8, onInput: () => redraw() });
  const minCtl = ui.number(controls, { key: 'min_val', label: 'min value', unit: 'a.u.', value: 0.0, onInput: () => redraw() });
  const maxCtl = ui.number(controls, { key: 'max_val', label: 'max value', unit: 'a.u.', value: 1.0, onInput: () => redraw() });
  const valueCtl = ui.slider(controls, { key: 'value', label: 'input value', unit: 'a.u.', min: 0, max: 1, step: 0.01, value: 0.5, onInput: () => redraw() });
  const deltaCtl = ui.slider(controls, { key: 'delta', label: 'distance Δ', unit: 'a.u.', min: 0, max: 0.5, step: 0.01, value: 0.02, onInput: () => redraw() });

  const bitsCanvas = ui.canvas(figures, { height: 90 });
  const overlapPlot = ui.plot(figures, { height: 220, xLabel: 'Δ (a.u.)', yLabel: 'overlap (bits)', xUnit: 'a.u.', yUnit: 'bits' });

  function opts() {
    return { size: sizeCtl.get(), n_active: Math.min(nActiveCtl.get(), sizeCtl.get()), min_val: minCtl.get(), max_val: maxCtl.get() };
  }

  function redraw() {
    const o = opts();
    const value = valueCtl.get();
    const delta = deltaCtl.get();
    const base = encodeScalar(value, o);
    const near = encodeScalar(value + delta, o);
    const overlap = base.overlap(near);

    const { canvas, ctx, width } = bitsCanvas;
    ctx.clearRect(0, 0, width, bitsCanvas.height);
    const pad = 8;
    const rowH = 18;
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const muted = ui.color('muted');
    const n = o.size;
    const gap = 2;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    const baseDense = base.toDense();
    const nearDense = near.toDense();
    for (let i = 0; i < n; i++) {
      const x = pad + i * (bw + gap);
      ctx.fillStyle = baseDense[i] ? c1 : muted;
      ctx.fillRect(x, pad, bw, rowH);
      ctx.fillStyle = nearDense[i] ? c2 : muted;
      ctx.fillRect(x, pad + rowH + 6, bw, rowH);
    }

    // Overlap as a function of distance.
    const xs = [];
    const ys = [];
    const steps = 50;
    for (let s = 0; s <= steps; s++) {
      const d = (delta * 2 * s) / steps; // 0 .. 2*delta
      xs.push(d);
      ys.push(base.overlap(encodeScalar(value + d, o)));
    }
    overlapPlot.draw([{ x: xs, y: ys, label: 'overlap', color: 'c1', style: 'step' }], {
      xRange: [0, 2 * delta],
      yRange: [0, o.n_active],
    });

    ui.note(
      notes,
      `<p>The scalar encoder maps a value to a contiguous block of <b>${o.n_active}</b> active bits whose center tracks the ` +
        `normalized value, wrapping around at the end. Value <b>${value.toFixed(2)}</b> and value <b>${(value + delta).toFixed(2)}</b> ` +
        `(Δ = ${delta.toFixed(2)} a.u.) share <b>${overlap}</b> bits.</p>` +
        `<p>Overlap falls as the two values drift apart: nearby values keep most of their bits, distant values share none. ` +
        `This is how a continuous quantity becomes a sparse code that HTM can compare by overlap. ` +
        `Values are clamped to [<b>${o.min_val}</b>, <b>${o.max_val}</b>]. All values are dimensionless (a.u.).</p>`
    );
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
    ui.code(notes, () =>
      `from conex import SDREncoder\n` +
      `enc = SDREncoder(size=${o.size}, n_active=${o.n_active}, encoder_type="scalar",\n` +
      `                 min_val=${o.min_val}, max_val=${o.max_val})\n` +
      `a = enc.encode_scalar(${value.toFixed(2)})\n` +
      `b = enc.encode_scalar(${(value + delta).toFixed(2)})\n` +
      `a.overlap(b)   # ${overlap} bits`
    );
  }

  redraw();
  ui.onTheme(redraw);
  return { destroy() {} };
}
