import { corticalColumnStructure } from '../js/models/tbt.js';

export const meta = {
  id: 'struct-column',
  title: 'Cortical column structure',
  conex: ['PredictiveCorticalColumn', 'Layer', 'ErrorUnit', 'PrecisionWeighting', 'FreeEnergyMinimization',
    'PredictionUnit', 'PredictiveCodingLearning', 'PredictiveSynapseInit', 'FeedforwardErrorSynapse',
    'FeedbackPredictionSynapse', 'PredictiveSynapseLearning'],
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

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// Which connection the pulse is on and its fraction along it, from the cycle
// position (0..1 over the whole CoNeX connection order).
export function pulsePosition(cycle, connOrder) {
  const n = connOrder.length;
  const pos = (cycle % 1) * n;
  const idx = Math.floor(pos) % n;
  const f = pos - Math.floor(pos);
  return { key: connOrder[idx], fraction: f };
}

// One-line narration driven by the model state.
export function columnNarration({ conn, connOrder }) {
  const i = connOrder.indexOf(conn);
  return `pulse ${i + 1} of ${connOrder.length}: ${conn.replace('_to_', ' → ')}`;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const sec = ui.section(controls, 'PredictiveCorticalColumn');
  const colSize = ui.number(sec, { key: 'columnSize', label: 'column_size', unit: 'neurons', value: 100 });

  ui.tryThis(figures, 'Try this: change <code>column_size</code> and watch the layer sizes scale (L2/3 is 1.5x, L5 0.8x, L6 0.5x).');

  // The hero canvas draws the stacked layers and the travelling pulse.
  const hero = ui.hero(figures, { height: 380 });
  ui.note(figures, '<em>Connections read from <code>PredictiveCorticalColumn._create_connections</code> in <code>conex/nn/structure/numenta/predictive_hierarchy.py</code>. The travelling pulse is <strong>schematic, not simulated</strong> &mdash; it only shows the order CoNeX applies the connections.</em>');
  const detail = ui.note(figures, '<em>Select a layer to see its neuron groups and behaviors.</em>');

  let structure = corticalColumnStructure(colSize.get());
  colSize.onInput(() => { structure = corticalColumnStructure(colSize.get()); render(); });

  // Pulses travel the connections in the order CoNeX builds them
  // (L4->L23, L23->L4, L23->L5, L6->L4). Schematic: no signal is simulated.
  const CONN_ORDER = ['L4_to_L23', 'L23_to_L4', 'L23_to_L5', 'L6_to_L4'];
  let connPaths = {}; // per connection: the 4 elbow points, for pulse positioning
  let layerBoxes = {}; // per layer: {x, y, w, h} in canvas px, for click hit-testing
  let selected = null;
  let currentCycle = 0;
  let mobileMode = false;

  // The drawing is laid out in a fixed 760x460 space and scaled to the canvas.
  const VW = 760, VH = 460;
  const BOX_W = 180, BOX_H = 64;
  const BOX_X = (VW - BOX_W) / 2;
  const GAP = (VH - 4 * BOX_H) / 5;
  const yOf = {};
  LAYER_ORDER.forEach((name, i) => { yOf[name] = GAP + i * (BOX_H + GAP); });
  const midY = (name) => yOf[name] + BOX_H / 2;
  const LANE = {
    'L4_to_L23': { x: BOX_X + BOX_W + 34, side: 'right' },
    'L23_to_L4': { x: BOX_X + BOX_W + 74, side: 'right' },
    'L23_to_L5': { x: BOX_X - 34, side: 'left' },
    'L6_to_L4': { x: BOX_X - 74, side: 'left' },
  };

  // Mobile layout: full-width layer cards stacked, then the connections in
  // CoNeX order below (the active one highlighted). 14 px labels.
  function renderMobile(ctx, width, height, col) {
    const { ink, muted, c1, c2, c3, c4, paper, layerColor, activeConn } = col;
    mobileMode = true;
    const pad = 12;
    const cardW = width - 2 * pad;
    const cardH = 46;
    const gap = 10;
    const topY = 8;
    layerBoxes = {};
    for (let i = 0; i < LAYER_ORDER.length; i++) {
      const name = LAYER_ORDER[i];
      const y = topY + i * (cardH + gap);
      layerBoxes[name] = { x: pad, y, w: cardW, h: cardH };
      const isSel = selected === name;
      ctx.fillStyle = paper;
      ctx.strokeStyle = layerColor[name];
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.fillRect(pad, y, cardW, cardH);
      ctx.strokeRect(pad, y, cardW, cardH);
      ctx.fillStyle = ink;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.font = '16px ui-serif, Georgia, serif';
      ctx.fillText(name === 'L23' ? 'L2/3' : name, pad + 12, y + cardH / 2);
      ctx.fillStyle = muted;
      ctx.font = '13px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${structure.layerSizes[name]} neurons`, pad + cardW - 12, y + cardH / 2);
    }
    // connections list below the cards
    const listTop = topY + LAYER_ORDER.length * (cardH + gap) + 6;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = muted;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('connections (CoNeX order):', pad, listTop);
    const connColor = { feedforward: c1, feedback: c4, lateral: c3 };
    for (let i = 0; i < CONN_ORDER.length; i++) {
      const key = CONN_ORDER[i];
      const conn = structure.connections.find((c) => `${c.from}_to_${c.to}` === key);
      const y = listTop + 18 + i * 18;
      const active = key === activeConn;
      ctx.fillStyle = active ? c2 : muted;
      ctx.beginPath();
      ctx.arc(pad + 5, y, active ? 4 : 2.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = active ? ink : muted;
      ctx.font = active ? 'bold 13px system-ui, sans-serif' : '13px system-ui, sans-serif';
      ctx.fillText(`${conn.from} → ${conn.to}  ${conn.label} (${conn.type})`, pad + 16, y);
    }
  }

  function render() {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c3 = ui.color('c3');
    const c4 = ui.color('c4');
    const paper = ui.color('paper');
    const layerColor = { L23: c1, L4: c2, L5: c3, L6: c4 };
    const { key: activeConn } = pulsePosition(currentCycle, CONN_ORDER);

    // Mobile: the elbow diagram is too small to read, so stack the layer cards
    // and list the connections (in CoNeX order) below, highlighting the pulse.
    if (width < 520) {
      renderMobile(ctx, width, height, { ink, muted, c1, c2, c3, c4, paper, layerColor, activeConn });
      hero.narration(columnNarration({ conn: activeConn, connOrder: CONN_ORDER }));
      return;
    }
    mobileMode = false;

    // scale the fixed drawing space uniformly to fit the canvas, centred
    const s = Math.min(width / VW, height / VH);
    const ox = (width - VW * s) / 2;
    const oy = (height - VH * s) / 2;
    const sx = (x) => ox + x * s;
    const sy = (y) => oy + y * s;

    // connections first (behind the boxes)
    for (const conn of structure.connections) {
      const key = `${conn.from}_to_${conn.to}`;
      const L = LANE[key] || LANE['L4_to_L23'];
      const yFrom = midY(conn.from);
      const yTo = midY(conn.to);
      const edge = L.side === 'right' ? BOX_X + BOX_W : BOX_X;
      const color = conn.type === 'feedback' ? c4 : conn.type === 'lateral' ? c3 : c1;
      const pts = [[edge, yFrom], [L.x, yFrom], [L.x, yTo], [edge, yTo]];
      connPaths[key] = pts;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
      for (let i = 1; i < 4; i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
      ctx.stroke();
      // arrowhead at the end
      const [ex, ey] = pts[3];
      const [px, py] = pts[2];
      const ang = Math.atan2(ey - py, ex - px);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(sx(ex), sy(ey));
      ctx.lineTo(sx(ex) - 8 * Math.cos(ang - 0.4), sy(ey) - 8 * Math.sin(ang - 0.4));
      ctx.lineTo(sx(ex) - 8 * Math.cos(ang + 0.4), sy(ey) - 8 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      // label beside the boxes, on its own lane
      const text = `${conn.label} (${conn.type})`;
      let ly = (yFrom + yTo) / 2;
      if (key === 'L4_to_L23') ly -= 11;
      if (key === 'L23_to_L4') ly += 11;
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = L.side === 'right' ? 'left' : 'right';
      ctx.textBaseline = 'middle';
      const lx = L.side === 'right' ? L.x + 6 : L.x - 6;
      ctx.fillStyle = muted;
      ctx.fillText(text, sx(lx), sy(ly));
    }

    // layer boxes (cards with a visible select affordance)
    for (const name of LAYER_ORDER) {
      const y = yOf[name];
      layerBoxes[name] = { x: BOX_X, y, w: BOX_W, h: BOX_H };
      const isSel = selected === name;
      ctx.fillStyle = paper;
      ctx.strokeStyle = layerColor[name];
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.fillRect(sx(BOX_X), sy(y), sx(BOX_W), sy(BOX_H));
      ctx.strokeRect(sx(BOX_X), sy(y), sx(BOX_W), sy(BOX_H));
      ctx.fillStyle = ink;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.font = '16px ui-serif, Georgia, serif';
      ctx.fillText(name === 'L23' ? 'L2/3' : name, sx(BOX_X + 12), sy(y + 26));
      ctx.fillStyle = muted;
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText(`${structure.layerSizes[name]} neurons`, sx(BOX_X + 12), sy(y + 46));
      // expand affordance
      ctx.fillStyle = layerColor[name];
      ctx.font = '18px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(isSel ? '−' : '+', sx(BOX_X + BOX_W - 12), sy(y + 26));
    }

    // pulse dot, moved by the player along the current connection
    const { key, fraction } = pulsePosition(currentCycle, CONN_ORDER);
    const pts = connPaths[key];
    if (pts) {
      const [px, py] = pointOnPath(pts, fraction);
      ctx.fillStyle = c2;
      ctx.strokeStyle = paper;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx(px), sy(py), 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
    hero.narration(columnNarration({ conn: key, connOrder: CONN_ORDER }));
  }

  // Position the pulse at fraction f (0..1) along a connection's elbow path.
  function pointOnPath(pts, f) {
    const segs = 3;
    const total = segs * f;
    const i = Math.min(segs - 1, Math.floor(total));
    const local = total - i;
    const a = pts[i], b = pts[i + 1];
    return [a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local];
  }

  function drawPulse(t) {
    currentCycle = (t / 100) % 1;
    render();
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

  // click a layer card to select it
  const canvas = hero.canvas;
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    let vx, vy;
    if (mobileMode) {
      // mobile boxes are in canvas pixels
      vx = ev.clientX - rect.left;
      vy = ev.clientY - rect.top;
    } else {
      const s = Math.min(hero.width / VW, hero.height / VH);
      const ox = (hero.width - VW * s) / 2;
      const oy = (hero.height - VH * s) / 2;
      vx = (ev.clientX - rect.left - ox) / s;
      vy = (ev.clientY - rect.top - oy) / s;
    }
    for (const name of LAYER_ORDER) {
      const b = layerBoxes[name];
      if (b && vx >= b.x && vx <= b.x + b.w && vy >= b.y && vy <= b.y + b.h) {
        selected = name;
        showDetail(name);
        render();
        return;
      }
    }
  });

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(render);
  render();
  ui.onTheme(render);

  // Player drives the pulse along the connections; time is a schematic
  // "cycle" counter, not a simulated signal.
  ui.player(figures, {
    duration: 100,
    unit: 'cycle',
    playSeconds: 6,
    loop: true,
    onFrame: drawPulse,
  });

  // --- notes ---
  ui.note(notes, `
    <p>A cortical column is a vertical stack of layers that all process the same
    patch of input. <code>PredictiveCorticalColumn</code> builds each layer as a
    <code>Layer</code> container holding one neuron group, and wires the layers
    together with plain <code>SynapseGroup</code> objects carrying the predictive
    coding behaviors. It is an independent class (not a
    <code>CorticalColumn</code> subclass) and takes any PyMonntorch
    <code>Network</code>. CoNeX also has the
    <code>CorticalColumn</code> / <code>CorticalLayer</code> /
    <code>CorticalLayerConnection</code> / <code>Neocortex</code> containers in
    <code>conex/nn/structure</code>, but this plate does not use them.</p>
    <p>The predictive coding column wires the layers into the canonical
    microcircuit: <strong>L4</strong> receives feedforward input and computes
    errors, <strong>L2/3</strong> generates predictions and sends them back down,
    <strong>L5</strong> produces output, and <strong>L6</strong> modulates
    precision. Only the connections the code actually creates are drawn here.</p>
    <p>Layer sizes scale with <code>column_size</code>: L2/3 is 1.5&times;, L5 is
    0.8&times; and L6 is 0.5&times; the base size. Select a layer card to see its
    neuron group and the behaviors it carries.</p>
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
