import { DendriticSegmentModel } from '../js/models/learning_numenta.js';

export const meta = {
  id: 'learn-segments',
  title: 'Dendritic segment learning',
  conex: ['ActiveDendriteComputation', 'DendriticSegment', 'TemporalMemory'],
  source: 'conex/behaviors/neurons/numenta/active_dendrites.py',
};

const NUM_STEPS = 10;
const N_SYN_PER_SEG = 4;
const N_PRE_CELLS = 8;

function mulberry(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- Parameters section (CoNeX defaults) ---
  const sec = ui.section(controls, 'DendriticSegment learning parameters');
  const permInc = ui.slider(sec, {
    key: 'permanenceIncrement', label: 'permanence_increment', unit: 'a.u.',
    min: 0.01, max: 0.25, step: 0.01, value: 0.1,
  });
  const permDec = ui.slider(sec, {
    key: 'permanenceDecrement', label: 'permanence_decrement', unit: 'a.u.',
    min: 0.005, max: 0.1, step: 0.005, value: 0.02,
  });
  const initPerm = ui.slider(sec, {
    key: 'initialPermanence', label: 'initial_permanence', unit: 'a.u.',
    min: 0.05, max: 0.5, step: 0.01, value: 0.21,
  });
  const connPerm = ui.slider(sec, {
    key: 'connectedPermanence', label: 'connected_permanence', unit: 'a.u.',
    min: 0.1, max: 0.6, step: 0.05, value: 0.2,
  });
  const actThresh = ui.slider(sec, {
    key: 'activationThreshold', label: 'activation_threshold', unit: 'synapses',
    min: 1, max: 4, step: 1, value: 2,
  });
  const learnThresh = ui.slider(sec, {
    key: 'learningThreshold', label: 'learning_threshold', unit: 'synapses',
    min: 1, max: 3, step: 1, value: 1,
  });
  const maxNew = ui.slider(sec, {
    key: 'maxNewSynapses', label: 'max_new_synapses', unit: 'synapses',
    min: 0, max: 4, step: 1, value: 2,
  });

  const patSec = ui.section(controls, 'Simulation setup');
  const seedCtrl = ui.number(patSec, { key: 'seed', label: 'seed', unit: '', value: 42 });

  // --- Figures ---
  const canvasFig = ui.canvas(figures, { height: 260 });
  const statusNote = ui.note(figures, '');
  const permPlot = ui.plot(figures, {
    height: 220,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'synapse permanence',
    yUnit: 'a.u.',
  });

  let currentStep = 0;
  let simData = null;

  function runSimulation() {
    const rng = mulberry(seedCtrl.get());
    const model = new DendriticSegmentModel({
      nCells: 1,
      nSegments: 2,
      nSynapsesPerSegment: N_SYN_PER_SEG,
      activationThreshold: actThresh.get(),
      learningThreshold: learnThresh.get(),
      permanenceIncrement: permInc.get(),
      permanenceDecrement: permDec.get(),
      initialPermanence: initPerm.get(),
      connectedPermanence: connPerm.get(),
      maxNewSynapses: maxNew.get(),
    });

    // Initialize initial segment targets and permanences
    // Segment 0 starts with 3 connected synapses and 1 empty slot
    model.setSegment(0, [0, 1, 2, -1], [0.30, 0.15, 0.25, 0.0]);
    // Segment 1 starts with 2 connected synapses and 2 empty slots
    model.setSegment(1, [3, 4, -1, -1], [0.25, 0.18, 0.0, 0.0]);

    // Pre-generate input patterns for NUM_STEPS
    const inputPatterns = [];
    for (let t = 0; t < NUM_STEPS; t++) {
      const active = new Array(N_PRE_CELLS).fill(false);
      // 2 to 3 active presynaptic neurons per step
      const nAct = 2 + Math.floor(rng() * 2);
      const chosen = new Set();
      while (chosen.size < nAct) {
        chosen.add(Math.floor(rng() * N_PRE_CELLS));
      }
      for (const idx of chosen) active[idx] = true;
      inputPatterns.push(active);
    }

    const steps = [];
    for (let t = 0; t < NUM_STEPS; t++) {
      const pre = inputPatterns[t];
      const seg0Before = model.getSegment(0);
      const seg1Before = model.getSegment(1);

      const actRes = model.computeActivation(pre);
      const actions = model.learnOnSegments({
        activeCells: [true],
        winnerCells: [true],
        presynapticActivity: pre,
        rng: { next: rng },
      });

      const seg0After = model.getSegment(0);
      const seg1After = model.getSegment(1);

      steps.push({
        step: t,
        presynapticActivity: pre,
        activation: actRes.activation,
        active: actRes.active,
        matching: actRes.matching,
        depolarization: actRes.depolarization[0],
        seg0Before,
        seg1Before,
        seg0After,
        seg1After,
        actions,
      });
    }

    simData = { steps, model };
  }

  function renderAt(step) {
    if (!simData) return;
    const clampedStep = Math.max(0, Math.min(step, NUM_STEPS - 1));
    currentStep = clampedStep;
    const sData = simData.steps[clampedStep];

    // 1. Draw canvas diagram
    drawCanvas(sData);

    // 2. Status note
    let actDetails = [];
    for (let s = 0; s < 2; s++) {
      const segAct = sData.actions[s] || [];
      const reinforced = segAct.filter((a) => a.action === 'reinforced').map((a) => `syn ${a.synapseIdx} (pre ${a.target})`);
      const punished = segAct.filter((a) => a.action === 'punished').map((a) => `syn ${a.synapseIdx} (pre ${a.target})`);
      const grown = segAct.filter((a) => a.action === 'grown').map((a) => `syn ${a.synapseIdx} (pre ${a.target})`);
      let summary = `<strong>Segment ${s}</strong> (active count = ${sData.activation[s]}, ${sData.matching[s] ? 'matched' : 'not matched'}): `;
      const parts = [];
      if (reinforced.length) parts.push(`<span style="color:var(--c5,#5b8a3c)">reinforced [${reinforced.join(', ')}]</span>`);
      if (punished.length) parts.push(`<span style="color:var(--c2,#d95b43)">punished [${punished.join(', ')}]</span>`);
      if (grown.length) parts.push(`<span style="color:var(--c1,#0f7b7f)">grown [${grown.join(', ')}]</span>`);
      if (!parts.length) parts.push('no weight changes');
      actDetails.push(summary + parts.join('; '));
    }

    const activePre = [];
    sData.presynapticActivity.forEach((isActive, idx) => {
      if (isActive) activePre.push(idx);
    });

    statusNote.innerHTML = `
      <div style="font-family:ui-monospace,monospace;font-size:0.85rem;line-height:1.4;">
        <div><strong>Step ${clampedStep}</strong>: Active presynaptic inputs = [${activePre.join(', ')}] &bull; Soma depolarization = ${(sData.depolarization * 100).toFixed(0)}%</div>
        <div>${actDetails.join('<br>')}</div>
      </div>
    `;

    // 3. Draw permanences plot over all steps with revealX
    const series = [];
    const colors = ['c1', 'c2', 'c3', 'c4'];
    for (let syn = 0; syn < N_SYN_PER_SEG; syn++) {
      const xs = [];
      const ys = [];
      for (let t = 0; t < NUM_STEPS; t++) {
        xs.push(t);
        ys.push(simData.steps[t].seg0After.permanences[syn]);
      }
      series.push({
        x: xs,
        y: ys,
        label: `Seg 0 Syn ${syn} (target ${sData.seg0After.targets[syn]})`,
        color: colors[syn % colors.length],
      });
    }

    permPlot.draw(series, {
      xRange: [0, NUM_STEPS - 1],
      yRange: [0, 1],
      revealX: clampedStep,
      hlines: [
        { y: connPerm.get(), label: `conn threshold (${connPerm.get()})` },
      ],
    });
  }

  function drawCanvas(sData) {
    const { canvas, ctx } = canvasFig;
    const w = canvas.width || 400;
    const h = canvas.height || 260;

    ctx.clearRect(0, 0, w, h);

    // Soma center at bottom
    const somaX = w / 2;
    const somaY = h - 35;
    const somaR = 24;

    ctx.save();

    // Draw soma
    ctx.beginPath();
    ctx.arc(somaX, somaY, somaR, 0, 2 * Math.PI);
    const depolVal = sData.depolarization;
    ctx.fillStyle = depolVal >= 1.0 ? ui.color('c2') : ui.color('c1');
    ctx.globalAlpha = 0.2 + 0.8 * depolVal;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = ui.color('ink');
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = ui.color('ink');
    ctx.font = '11px ui-sans-serif, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Soma', somaX, somaY - 4);
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(`${(depolVal * 100).toFixed(0)}% depol`, somaX, somaY + 8);

    // Draw 2 dendritic branches (Segment 0 left, Segment 1 right)
    const branches = [
      { segIdx: 0, startX: somaX - 10, startY: somaY - somaR, endX: w * 0.25, endY: 50 },
      { segIdx: 1, startX: somaX + 10, startY: somaY - somaR, endX: w * 0.75, endY: 50 },
    ];

    branches.forEach((b) => {
      const isMatching = sData.matching[b.segIdx];
      const isActive = sData.active[b.segIdx];

      // Trunk line
      ctx.beginPath();
      ctx.moveTo(b.startX, b.startY);
      ctx.quadraticCurveTo(
        b.startX + (b.endX - b.startX) * 0.3,
        b.startY - 40,
        b.endX,
        b.endY,
      );
      ctx.strokeStyle = isActive ? ui.color('c2') : isMatching ? ui.color('c3') : ui.color('muted');
      ctx.lineWidth = isActive ? 4 : isMatching ? 3 : 2;
      ctx.stroke();

      // Branch label
      ctx.fillStyle = ui.color('ink');
      ctx.font = 'bold 11px ui-sans-serif, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Segment ${b.segIdx} (${sData.activation[b.segIdx]} active)`,
        b.endX,
        b.endY - 14,
      );

      // Draw 4 synapses along branch
      const segActions = sData.actions[b.segIdx] || [];
      for (let i = 0; i < N_SYN_PER_SEG; i++) {
        const tFrac = 0.25 + i * 0.22;
        // Point along quadratic curve
        const qx = (1 - tFrac) * (1 - tFrac) * b.startX + 2 * (1 - tFrac) * tFrac * (b.startX + (b.endX - b.startX) * 0.3) + tFrac * tFrac * b.endX;
        const qy = (1 - tFrac) * (1 - tFrac) * b.startY + 2 * (1 - tFrac) * tFrac * (b.startY - 40) + tFrac * tFrac * b.endY;

        const action = segActions[i] || {};
        const tgt = action.target !== undefined ? action.target : -1;
        const permAfter = action.permAfter !== undefined ? action.permAfter : 0.0;
        const isConn = permAfter >= connPerm.get();

        // Synapse circle
        ctx.beginPath();
        ctx.arc(qx, qy, 8, 0, 2 * Math.PI);

        let synColor = ui.color('paper');
        if (action.action === 'reinforced') synColor = ui.color('c5');
        else if (action.action === 'punished') synColor = ui.color('c2');
        else if (action.action === 'grown') synColor = ui.color('c1');
        else if (isConn) synColor = ui.color('c3');

        ctx.fillStyle = synColor;
        ctx.fill();
        ctx.strokeStyle = isConn ? ui.color('ink') : ui.color('muted');
        ctx.lineWidth = isConn ? 2 : 1;
        ctx.stroke();

        // Target label
        ctx.fillStyle = ui.color('ink');
        ctx.font = '9px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tgt >= 0 ? `${tgt}` : 'ø', qx, qy);

        // Permanence readout to side
        ctx.font = '8px ui-monospace, monospace';
        ctx.textAlign = b.segIdx === 0 ? 'right' : 'left';
        const textX = b.segIdx === 0 ? qx - 12 : qx + 12;
        ctx.fillText(`p=${permAfter.toFixed(2)}`, textX, qy);
      }
    });

    ctx.restore();
  }

  // --- Animation / Player ---
  let player = null;
  if (typeof ui.player === 'function') {
    player = ui.player(figures, {
      duration: NUM_STEPS - 1,
      unit: 'step',
      playSeconds: 6,
      loop: true,
      onFrame: (t) => {
        renderAt(Math.round(t));
      },
    });
  }

  function rebuild() {
    runSimulation();
    renderAt(currentStep);
  }

  [permInc, permDec, initPerm, connPerm, actThresh, learnThresh, maxNew, seedCtrl].forEach((c) => {
    c.onInput(rebuild);
  });

  canvasFig.onResize(() => {
    renderAt(currentStep);
  });
  ui.onTheme(() => {
    renderAt(currentStep);
  });

  rebuild();

  // --- Notes section ---
  ui.note(notes, `
    <p>In neocortical pyramidal neurons, distal dendritic branches process contextual and predictive input.
    When coincident presynaptic spikes activate a cluster of synapses on the same dendritic segment, they evoke an
    all-or-none NMDA plateau potential that depolarizes the soma. This subthreshold depolarization does not initiate
    a somatic spike immediately, but primes the neuron to fire faster than unpredicted neighbors upon feedforward drive.</p>
    <p>Segment learning in CoNeX (implemented in <code>DendriticSegment.learn_on_segments</code> and
    <code>TemporalMemory.adapt_segment</code>) enforces Hebbian structural plasticity:
    synapses that successfully predicted activation are <strong>reinforced</strong> (+<code>permanence_increment</code>),
    while co-allocated synapses from inactive presynaptic cells are <strong>punished</strong> (-<code>permanence_decrement</code>).
    Segments with room grow new synapses to active inputs with <code>initial_permanence</code>.</p>
    <p>When a segment causes a false prediction (the cell was predicted to fire but feedforward drive never arrived),
    <code>punish_segments</code> decrements the active synapses that produced the false alarm, pruning non-predictive connections.</p>
  `);

  ui.equation(notes, `
    <p>Segment activation: <var>a</var><sub>s</sub> = &sum;<sub>j</sub> [ <var>p</var><sub>s,j</sub> &ge; <var>connected_perm</var> ] &middot; [ <var>x</var><sub>j</sub> = 1 ]</p>
    <p>Matching segment: <var>a</var><sub>s</sub> &ge; <var>learning_threshold</var>, &nbsp; Active segment: <var>a</var><sub>s</sub> &ge; <var>activation_threshold</var></p>
    <p>Depolarization: <var>V</var><sub>depol</sub> = min(1.0, max<sub>s</sub>(<var>a</var><sub>s</sub>) / <var>activation_threshold</var>)</p>
    <p>Reinforce active: <var>p</var><sub>s,j</sub> &larr; clamp(<var>p</var><sub>s,j</sub> + <var>perm_inc</var>, 0, 1)</p>
    <p>Punish inactive: <var>p</var><sub>s,j</sub> &larr; clamp(<var>p</var><sub>s,j</sub> &minus; <var>perm_dec</var>, 0, 1)</p>
  `);

  ui.note(notes, `
    <p><strong>Constants &amp; Sources:</strong> Parameters follow Hawkins et al. (2016),
    <em>"Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex"</em>,
    and the CoNeX defaults in <code>DendriticSegmentConfig</code> and <code>TemporalMemoryConfig</code>.</p>
    <p><strong>ML analogy:</strong> Each dendritic segment functions like an independent binary classifier
    with bounded structural weights [0, 1] learning an online linear decision surface for a specific contextual pattern.</p>
    <p><strong>Where the analogy breaks:</strong> Artificial neural network units sum all inputs into a single global
    activation and update via backpropagation. In pyramidal neurons, hundreds of segments operate as independent
    non-linear subunits in parallel (an OR of ANDs), and learning operates through localized structural synaptogenesis
    without modifying feedforward receptive fields.</p>
  `);

  // --- CoNeX code box ---
  ui.code(notes, () => `import torch
from pymonntorch import Network, NeuronGroup
from conex import DendriticSegment, DendriticSegmentConfig, TimeResolution

config = DendriticSegmentConfig(
    n_synapses_per_segment=${N_SYN_PER_SEG},
    activation_threshold=${actThresh.get()},
    learning_threshold=${learnThresh.get()},
    permanence_increment=${permInc.get()},
    permanence_decrement=${permDec.get()},
    initial_permanence=${initPerm.get()},
    connected_permanence=${connPerm.get()},
    max_new_synapses=${maxNew.get()},
)
net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
neurons = NeuronGroup(
    net=net,
    size=1,
    behavior={10: DendriticSegment(n_segments=2, n_cells=1, config=config)},
    tag="pyramidal_cell",
)
net.initialize()
`);

  return {
    destroy() {
      if (player && typeof player.pause === 'function') player.pause();
    },
  };
}
