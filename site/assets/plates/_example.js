// Example plate: shows ui.js working. Not listed in any catalog;
// plate.html?id=_example loads it as a special case.
export const meta = {
  id: '_example',
  title: 'Example plate',
  conex: [],
  source: 'site/assets/plates/_example.js',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'Wave');

  const DT = 1; // ms of model time per step
  const WINDOW = 200; // ms of model time kept on screen
  let t = 0; // model time of the newest sample
  const ring = []; // [t, v] samples of the last WINDOW ms

  // One dt of the model: evaluate v(t) and keep the rolling window.
  function step() {
    t += DT;
    const v = amp.get() * Math.sin(2 * Math.PI * freq.get() * (t / 1000) + phase.get());
    ring.push([t, v]);
    while (ring.length && ring[0][0] < t - WINDOW) ring.shift();
  }

  function redraw() {
    const x = ring.map((p) => p[0]);
    const y = ring.map((p) => p[1]);
    fig.draw([{ x, y, label: 'v(t)', color: 'c1', style: 'line' }],
      { xRange: [t - WINDOW, t] });
  }

  const amp = ui.slider(sec, {
    key: 'amp', label: 'Amplitude', unit: 'a.u.',
    min: 0.1, max: 3, step: 0.1, value: 1,
  });
  const freq = ui.slider(sec, {
    key: 'freq', label: 'Frequency', unit: 'Hz',
    min: 0.1, max: 5, step: 0.1, value: 1,
  });
  const phase = ui.slider(sec, {
    key: 'phase', label: 'Phase', unit: 'rad',
    min: 0, max: 6.28, step: 0.01, value: 0,
  });

  const fig = ui.plot(figures, {
    height: 260, xLabel: 'time', yLabel: 'v', xUnit: 'ms', yUnit: 'a.u.',
  });

  // The live runner steps the model each frame; the plot scrolls a 200 ms
  // window. Slider changes apply on the next step without a reset.
  ui.live(sec, {
    dt: DT, unit: 'ms', modelPerSecond: 100, windowSize: WINDOW,
    step,
    reset: () => { t = 0; ring.length = 0; },
    onFrame: redraw,
  });
  // fill the window once so the figure is not empty before the first frame
  for (let i = 0; i < WINDOW / DT; i++) step();
  redraw();

  // 8x8 heatmap of a seeded random field
  const R = ui.rng(42);
  const heat = [];
  for (let i = 0; i < 64; i++) heat.push(R.next());
  const heatFig = ui.heatmap(figures, { rows: 8, cols: 8, label: 'field (a.u.)', height: 200 });
  heatFig.draw(heat, { min: 0, max: 1 });

  // raster of 20 seeded spikes across 5 neurons over 100 ms
  const R2 = ui.rng(7);
  const spikes = [];
  for (let i = 0; i < 20; i++) {
    spikes.push([R2.next() * 100, Math.floor(R2.next() * 5)]);
  }
  const rasFig = ui.raster(figures, { height: 160, xLabel: 'time', xUnit: 'ms' });
  rasFig.draw(spikes, 5, 100);

  // free canvas: a circle whose radius follows the amplitude slider
  const cv = ui.canvas(figures, { height: 160 });
  function drawCircle() {
    const { ctx, width, height } = cv;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = ui.color('c2');
    ctx.lineWidth = 2;
    const r = 10 + amp.get() * 10; // radius grows with the amplitude slider
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  drawCircle();
  cv.onResize(drawCircle);
  amp.onInput(drawCircle);

  ui.note(notes, 'This plate is a demo of the shared <code>ui.js</code> API: ' +
    'three sliders, a live plot with a rolling 200 ms window, and URL-hash ' +
    'state. Change a slider and the value appears in the address bar; reload ' +
    'and it is restored. The wave is stepped live, so a slider move changes ' +
    'the trace without restarting it.');
  ui.equation(notes, '<var>v</var>(<var>t</var>) = <var>A</var> sin(2&#960;<var>f</var><var>t</var> + <var>&phi;</var>)');
  ui.code(notes, () =>
    `# demo only, not CoNeX API\n` +
    `A = ${amp.get()}\n` +
    `f = ${freq.get()}  # Hz\n` +
    `phi = ${phase.get()}  # rad\n`
  );

  return { destroy() {} };
}
