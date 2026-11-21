import { corticalColumnStructure } from '../js/models/tbt.js';

export const meta = {
  id: 'struct-column',
  title: 'Cortical column structure',
  conex: ['CorticalColumn', 'Layer', 'Neocortex', 'CorticalLayerConnection'],
  source: 'conex/nn/structure/numenta/predictive_hierarchy.py',
};

// Behaviors each layer carries, exactly as PredictiveCorticalColumn._build wires them.
const LAYER_BEHAVIORS = {
  L4: ['ErrorUnit', 'PrecisionWeighting', 'FreeEnergyMinimization'],
  L23: ['PredictionUnit', 'ErrorUnit', 'PredictiveCodingLearning'],
  L5: ['PredictionUnit', 'PredictiveCodingLearning'],
  L6: ['PrecisionWeighting'],
};
const LAYER_ROLE = {
  L4: 'Granular input layer: receives feedforward input and computes prediction errors.',
  L23: 'Supragranular layer: generates predictions and sends them back down.',
  L5: 'Infragranular output layer: motor / output predictions.',
  L6: 'Precision modulation layer: modulates the precision of lower-level errors.',
};
const LAYER_ORDER = ['L23', 'L4', 'L5', 'L6']; // top to bottom in the drawing

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'PredictiveCorticalColumn');
  const colSize = ui.number(sec, { key: 'columnSize', label: 'column_size', unit: 'neurons', value: 100 });

  const svgHost = ui.note(figures, '');
  ui.note(figures, '<em>Connections read from <code>PredictiveCorticalColumn._create_connections</code> in <code>conex/nn/structure/numenta/predictive_hierarchy.py</code>.</em>');
  const detail = ui.note(figures, '<em>click a layer to see its neuron groups and behaviors</em>');

  let structure = corticalColumnStructure(colSize.get());
  colSize.onInput(() => { structure = corticalColumnStructure(colSize.get()); render(); });

  const SVGNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function render() {
    const W = 560;
    const H = 460;
    const boxW = 180;
    const boxH = 64;
    const x = (W - boxW) / 2;
    const gap = (H - 4 * boxH) / 5;
    const yOf = {};
    LAYER_ORDER.forEach((name, i) => { yOf[name] = gap + i * (boxH + gap); });

    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H });
    svg.style.maxWidth = W + 'px';
    svg.style.display = 'block';
    svg.style.margin = '0 auto';

    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const c4 = ui.color('c4');
    const paper = ui.color('paper');
    const layerColor = { L23: c1, L4: c2, L5: c3, L6: c4 };

    // connections first (behind the boxes)
    for (const conn of structure.connections) {
      const from = yOf[conn.from];
      const to = yOf[conn.to];
      const cx = x + boxW / 2;
      const down = to > from;
      const y1 = down ? from + boxH : from;
      const y2 = down ? to : to + boxH;
      const color = conn.type === 'feedback' ? c4 : conn.type === 'lateral' ? c3 : c1;
      const path = svgEl('path', {
        d: `M ${cx} ${y1} L ${cx} ${y2}`,
        stroke: color,
        'stroke-width': 2,
        fill: 'none',
        'marker-end': `url(#arrow-${conn.type})`,
      });
      svg.appendChild(path);
      // label
      const label = svgEl('text', {
        x: cx + 8,
        y: (y1 + y2) / 2,
        fill: muted,
        'font-size': 11,
        'font-family': 'system-ui, sans-serif',
      });
      label.textContent = `${conn.label} (${conn.type})`;
      svg.appendChild(label);
    }

    // arrowhead markers, one per connection type
    const defs = svgEl('defs', {});
    for (const type of ['feedforward', 'feedback', 'lateral']) {
      const color = type === 'feedback' ? c4 : type === 'lateral' ? c3 : c1;
      const marker = svgEl('marker', {
        id: `arrow-${type}`, viewBox: '0 0 10 10', refX: 8, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse',
      });
      marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }));
      defs.appendChild(marker);
    }
    svg.appendChild(defs);

    // layer boxes
    for (const name of LAYER_ORDER) {
      const y = yOf[name];
      const g = svgEl('g', { cursor: 'pointer' });
      const rect = svgEl('rect', {
        x, y, width: boxW, height: boxH, rx: 6,
        fill: paper, stroke: layerColor[name], 'stroke-width': 2,
      });
      const title = svgEl('text', {
        x: x + 12, y: y + 26, fill: ink, 'font-size': 16,
        'font-family': 'ui-serif, Georgia, serif',
      });
      title.textContent = name === 'L23' ? 'L2/3' : name;
      const size = svgEl('text', {
        x: x + 12, y: y + 46, fill: muted, 'font-size': 12,
        'font-family': 'ui-monospace, monospace',
      });
      size.textContent = `${structure.layerSizes[name]} neurons`;
      g.appendChild(rect);
      g.appendChild(title);
      g.appendChild(size);
      g.addEventListener('click', () => showDetail(name));
      svg.appendChild(g);
    }

    svgHost.innerHTML = '';
    svgHost.appendChild(svg);
  }

  function showDetail(name) {
    const behaviors = LAYER_BEHAVIORS[name];
    detail.innerHTML = `
      <p><strong>${name === 'L23' ? 'L2/3' : name}</strong> &mdash; ${structure.layerSizes[name]} neurons.
      ${LAYER_ROLE[name]}</p>
      <p>Neuron group: <code>${name}_pop</code> (a single population, wrapped in a
      <code>Layer</code>).</p>
      <p>Behaviors: ${behaviors.map((b) => `<code>${b}</code>`).join(', ')}.</p>
    `;
  }

  render();
  ui.onTheme(render);

  // --- notes ---
  ui.note(notes, `
    <p>A cortical column is a vertical stack of layers that all process the same
    patch of input. CoNeX builds the column from <code>Layer</code> containers
    (each holding one or more neuron groups) joined by
    <code>CorticalLayerConnection</code> objects, all inside a
    <code>CorticalColumn</code> container, which itself lives in a
    <code>Neocortex</code> network.</p>
    <p>The predictive coding column wires the layers into the canonical
    microcircuit: <strong>L4</strong> receives feedforward input and computes
    errors, <strong>L2/3</strong> generates predictions and sends them back down,
    <strong>L5</strong> produces output, and <strong>L6</strong> modulates
    precision. Only the connections the code actually creates are drawn here.</p>
    <p>Layer sizes scale with <code>column_size</code>: L2/3 is 1.5&times;, L5 is
    0.8&times; and L6 is 0.5&times; the base size.</p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> a column is like a small residual block with
    a skip connection &mdash; feedforward (error) going up, feedback (prediction)
    coming back down, and a lateral precision gate. The analogy breaks because the
    layers are fixed biological structures with specific neuron types and
    synapse dynamics, not interchangeable learned layers, and the "residual" is a
    prediction error, not an identity shortcut.</p>
  `);

  ui.code(notes, () => `
from conex import (
    Neocortex, CorticalColumn, CorticalLayer, CorticalLayerConnection,
    PredictiveCorticalColumn,
)

net = Neocortex(dt=1.0)
# the predictive coding column builds L4, L2/3, L5, L6 and their
# feedforward / feedback / lateral connections automatically:
column = PredictiveCorticalColumn(net, column_size=${colSize.get()}, name="col")
print(column.layer_sizes)
print(sorted(column.synapses.keys()))
`);

  return { destroy() {} };
}
