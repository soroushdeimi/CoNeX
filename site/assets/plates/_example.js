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

  function redraw() {
    const N = 200;
    const x = [], y = [];
    for (let i = 0; i < N; i++) {
      const t = (i / (N - 1)) * 100; // 0..100 ms
      x.push(t);
      y.push(amp.get() * Math.sin(2 * Math.PI * freq.get() * (t / 1000) + phase.get()));
    }
    fig.draw([{ x, y, label: 'v(t)', color: 'c1', style: 'line' }]);
  }

  const amp = ui.slider(sec, {
    key: 'amp', label: 'Amplitude', unit: 'a.u.',
    min: 0.1, max: 3, step: 0.1, value: 1, onInput: redraw,
  });
  const freq = ui.slider(sec, {
    key: 'freq', label: 'Frequency', unit: 'Hz',
    min: 0.1, max: 5, step: 0.1, value: 1, onInput: redraw,
  });
  const phase = ui.slider(sec, {
    key: 'phase', label: 'Phase', unit: 'rad',
    min: 0, max: 6.28, step: 0.01, value: 0, onInput: redraw,
  });

  const fig = ui.plot(figures, {
    height: 260, xLabel: 'time', yLabel: 'v', xUnit: 'ms', yUnit: 'a.u.',
  });
  redraw();

  ui.note(notes, 'This plate is a demo of the shared <code>ui.js</code> API: ' +
    'three sliders, a live plot, and URL-hash state. Change a slider and the ' +
    'value appears in the address bar; reload and it is restored.');
  ui.equation(notes, '<var>v</var>(<var>t</var>) = <var>A</var> sin(2&#960;<var>f</var><var>t</var> + <var>&phi;</var>)');
  ui.code(notes, () =>
    `# demo only, not CoNeX API\n` +
    `A = ${amp.get()}\n` +
    `f = ${freq.get()}  # Hz\n` +
    `phi = ${phase.get()}  # rad\n`
  );

  return { destroy() {} };
}
