import { encodeScalar } from '../js/models/htm.js';

export const meta = {
  id: 'htm-encoder',
  title: 'Scalar SDR encoder',
  conex: ['SDREncoder'],
  source: 'conex/behaviors/neurons/numenta/sdr.py',
};

// Pure hero state for the encoder plate. The dial needle sits at the
// normalized sweep value; the active block slides across the bit strip and the
// bits shared with the reference are highlighted. t is the player time, in
// value units (a.u.).
export function encoderHeroState(run, t) {
  const { o, ref } = run;
  const v = o.min_val + t;
  const cur = encodeScalar(v, o);
  const refDense = run.base.toDense();
  const curDense = cur.toDense();
  const overlap = new Array(o.size).fill(0);
  let nOverlap = 0;
  for (let i = 0; i < o.size; i++) {
    if (refDense[i] && curDense[i]) {
      overlap[i] = 1;
      nOverlap++;
    }
  }
  const norm = Math.max(0, Math.min(1, (v - o.min_val) / (o.max_val - o.min_val)));
  const narration =
    `value ${v.toFixed(2)} a.u.: block of ${o.n_active} bits, ` +
    `overlap with reference (${ref.toFixed(2)}) = ${nOverlap} bits`;
  return { v, norm, refDense, curDense, overlap, nOverlap, narration };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sizeCtl = ui.number(controls, { key: 'size', label: 'SDR size', unit: 'bits', value: 64, onInput: () => recompute() });
  const nActiveCtl = ui.number(controls, { key: 'n_active', label: 'active bits', unit: 'bits', value: 8, onInput: () => recompute() });
  const minCtl = ui.number(controls, { key: 'min_val', label: 'min value', unit: 'a.u.', value: 0.0, onInput: () => recompute() });
  const maxCtl = ui.number(controls, { key: 'max_val', label: 'max value', unit: 'a.u.', value: 1.0, onInput: () => recompute() });
  const valueCtl = ui.slider(controls, { key: 'value', label: 'reference value', unit: 'a.u.', min: 0, max: 1, step: 0.01, value: 0.5, onInput: () => recompute() });
  const deltaCtl = ui.slider(controls, { key: 'delta', label: 'distance Δ', unit: 'a.u.', min: 0, max: 0.5, step: 0.01, value: 0.02, onInput: () => recompute() });

  // Hero: a value dial with a sweeping needle above the sliding bit strip.
  const hero = ui.hero(figures, { height: 150 });
  const heroCap = { set(text) { hero.narration(text); } };
  const overlapPlot = ui.plot(figures, { height: 220, xLabel: 'input value', yLabel: 'overlap', xUnit: 'a.u.', yUnit: 'bits' });
  const readout = document.createElement('div');
  readout.className = 'ui-note';
  figures.appendChild(readout);

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
    const { o, ref, delta, xs, ys } = run;
    const v = o.min_val + t;
    drawHero();
    overlapPlot.draw([{ x: xs, y: ys, label: 'overlap', color: 'c1', style: 'step' }], {
      xRange: [o.min_val, o.max_val],
      yRange: [0, o.n_active],
      revealX: v,
      // The reference and ref+Δ values live in the caption below (the two
      // labels overprint near the staircase peak); the lines stay as guides.
      vlines: [{ x: ref, label: '' }, { x: ref + delta, label: '' }],
    });
    const ov = run.base.overlap(encodeScalar(ref + delta, o));
    readout.textContent =
      `reference ${ref.toFixed(2)} a.u. · ref+Δ ${(ref + delta).toFixed(2)} a.u. · ` +
      `overlap(ref, ref+Δ) = ${ov} bits`;
  }

  // Hero: a value dial (needle at the normalized sweep value) above two rows
  // of bits — the reference block (teal) and the sweeping block (coral), with
  // the shared bits highlighted in ochre.
  function drawHero() {
    const st = encoderHeroState(run, lastT);
    const { ctx, width, height } = hero;
    const pad = 8;
    const rowH = 18;
    const dialH = 40;
    const n = run.o.size;
    const gap = 2;
    const bw = (width - 2 * pad - gap * (n - 1)) / n;
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const muted = ui.color('muted');
    const ink = ui.color('ink');
    ctx.clearRect(0, 0, width, height);

    // Dial: a horizontal track with a needle at the normalized value.
    const trackY = pad + dialH / 2;
    ctx.strokeStyle = muted;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pad, trackY);
    ctx.lineTo(width - pad, trackY);
    ctx.stroke();
    const nx = pad + st.norm * (width - 2 * pad);
    ctx.strokeStyle = ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(nx, trackY - 12);
    ctx.lineTo(nx, trackY + 12);
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(st.v.toFixed(2), nx, trackY - 14);

    // Bit rows below the dial.
    const topY = pad + dialH;
    for (let i = 0; i < n; i++) {
      const x = pad + i * (bw + gap);
      ctx.fillStyle = st.refDense[i] ? c1 : muted;
      ctx.fillRect(x, topY, bw, rowH);
      ctx.fillStyle = st.curDense[i] ? c2 : muted;
      ctx.fillRect(x, topY + rowH + 6, bw, rowH);
      if (st.overlap[i]) {
        ctx.strokeStyle = c3;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, topY + rowH + 7, bw - 2, rowH - 2);
      }
    }
    heroCap.set(st.narration);
  }
  // The hero canvas resets its backing store on resize and theme change, so
  // register the draw via onResize (called after layout and on every resize).
  hero.onResize(() => { if (run) drawHero(); });

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
        `normalized value, linearly across the range. The animation sweeps the input value across ` +
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
        `The analogy breaks because the encoding is linear, not circular: the center is ` +
        `⌊norm(v)·(n−w)⌋, so the min and max values have zero overlap and the two ends of the range are not neighbors.</p>`
    );
    dynNotes = document.createElement('div');
    notes.appendChild(dynNotes);
  }

  staticNotes();
  recompute();
  ui.onTheme(() => frame(lastT));
  return { destroy() { if (player) player.pause(); } };
}
