import { DendriticSegmentModel } from '../js/models/learning_numenta.js';
import { ease } from '../js/core.js';

export const meta = {
  id: 'learn-segments',
  title: 'Dendritic segment learning',
  conex: ['ActiveDendriteComputation', 'DendriticSegment', 'SegmentStorage', 'TemporalMemory'],
  source: 'conex/behaviors/neurons/numenta/active_dendrites.py',
};

const NUM_STEPS = 10;
const N_SYN_PER_SEG = 20;
const N_PRE_CELLS = 40;

// Bead radius (px) for a synapse: permanence in [0,1] maps to [2,8] px.
export function beadRadius(perm) {
  const p = Math.max(0, Math.min(1, perm));
  return 2 + 6 * p;
}

// Pure hero state for the dendritic segment plate. For each segment and
// synapse it eases the permanence (and thus the bead radius) between the
// before and after of this step, flags the learning action, and marks whether
// the presynaptic target is active (so a pulse travels up the branch).
export function segHeroState(run, t) {
  const N = run.steps.length;
  const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
  const f = ease(t - idx);
  const st = run.steps[idx];
  const segs = [];
  for (let s = 0; s < 2; s++) {
    const before = s === 0 ? st.seg0Before : st.seg1Before;
    const after = s === 0 ? st.seg0After : st.seg1After;
    const actions = st.actions[s] || [];
    const syn = [];
    for (let i = 0; i < before.permanences.length; i++) {
      const perm = before.permanences[i] * (1 - f) + after.permanences[i] * f;
      const target = after.targets[i];
      const action = actions[i] ? actions[i].action : 'unchanged';
      const active = target >= 0 && Boolean(st.presynapticActivity[target]);
      syn.push({ target, perm, radius: beadRadius(perm), action, active });
    }
    segs.push({ activation: st.activation[s], active: st.active[s], matching: st.matching[s], syn });
  }
  const nReinforced = segs[0].syn.filter((x) => x.action === 'reinforced').length;
  const nPunished = segs[0].syn.filter((x) => x.action === 'punished').length;
  const nGrown = segs[0].syn.filter((x) => x.action === 'grown').length;
  const narration =
    `step ${idx + 1}/${N}: segment 0 matched=${st.matching[0] ? 'yes' : 'no'} — ` +
    `${nReinforced} reinforced, ${nPunished} punished, ${nGrown} grown; ` +
    `depolarization ${(st.depolarization * 100).toFixed(0)}%`;
  return { idx, f, segs, depol: st.depolarization, narration };
}

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

  // --- Parameters section (CoNeX DendriticSegmentConfig defaults) ---
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
    min: 0.05, max: 0.8, step: 0.05, value: 0.2,
  });
  const actThresh = ui.slider(sec, {
    key: 'activationThreshold', label: 'activation_threshold', unit: 'synapses',
    min: 1, max: 20, step: 1, value: 10,
  });
  const learnThresh = ui.slider(sec, {
    key: 'learningThreshold', label: 'learning_threshold', unit: 'synapses',
    min: 1, max: 20, step: 1, value: 8,
  });
  const maxNew = ui.slider(sec, {
    key: 'maxNewSynapses', label: 'max_new_synapses', unit: 'synapses',
    min: 0, max: 10, step: 1, value: 5,
  });

  const patSec = ui.section(controls, 'Simulation setup');
  const contextOverlap = ui.slider(patSec, {
    key: 'contextOverlap', label: 'context overlap with segment 0', unit: 'cells',
    min: 0, max: 15, step: 1, value: 10,
  });
  const seedCtrl = ui.number(patSec, { key: 'seed', label: 'seed', unit: '', value: 42 });

  // --- Figures ---
  const canvasFig = ui.hero(figures, { height: 270 });
  const heroCap = { set(text) { canvasFig.narration(text); } };
  const depolCaption = ui.note(figures, '<em>Depolarization is computed as max segment activity / activation_threshold, capped at 1, as in compute_depolarization (active_dendrites.py:181).</em>');
  // The per-step details live in a collapsible <details> so they do not
  // duplicate the hero narration line.
  const statusNote = document.createElement('details');
  statusNote.className = 'ui-note';
  statusNote.style.cssText = 'font-size:0.85rem;';
  const statusSummary = document.createElement('summary');
  statusSummary.textContent = 'step details';
  statusNote.appendChild(statusSummary);
  const statusBody = document.createElement('div');
  statusBody.style.cssText = 'font-family:ui-monospace,monospace;line-height:1.4;margin-top:0.3rem;';
  statusNote.appendChild(statusBody);
  figures.appendChild(statusNote);
  const permPlot = ui.plot(figures, {
    height: 220,
    xLabel: 'learning step',
    xUnit: 'step',
    yLabel: 'synapse permanence',
    yUnit: 'a.u.',
  });
  const legendNote = ui.note(figures, '<em>Permanence trajectories for 6 representative synapses on Segment 0: active inputs (Syn 0, 2, 5: reinforced), inactive inputs (Syn 10, 12: punished), and newly grown synapse (Syn 15).</em>');

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

    // Initialize 2 segments:
    // Segment 0: targets 0..14 initialized to 0.25 (connected >= 0.2) + 5 empty slots (-1)
    // Segment 1: targets 15..29 initialized to 0.25 (connected >= 0.2) + 5 empty slots (-1)
    const seg0Targets = [];
    const seg0Perms = [];
    const seg1Targets = [];
    const seg1Perms = [];

    for (let i = 0; i < N_SYN_PER_SEG; i++) {
      if (i < 15) {
        seg0Targets.push(i);
        seg0Perms.push(0.25);
        seg1Targets.push(15 + i);
        seg1Perms.push(0.25);
      } else {
        seg0Targets.push(-1);
        seg0Perms.push(0.0);
        seg1Targets.push(-1);
        seg1Perms.push(0.0);
      }
    }

    model.setSegment(0, seg0Targets, seg0Perms);
    model.setSegment(1, seg1Targets, seg1Perms);

    const overlap = contextOverlap.get();

    // Generate input patterns for NUM_STEPS:
    // Segment 0 sees `overlap` active presynaptic cells (targets 0..overlap-1)
    // Plus 5 novel presynaptic cells (35..39) active to trigger synaptogenesis into empty slots
    // Segment 1 sees 0 active cells
    const inputPatterns = [];
    for (let t = 0; t < NUM_STEPS; t++) {
      const active = new Array(N_PRE_CELLS).fill(false);
      // Active targets matching Segment 0
      for (let i = 0; i < overlap; i++) {
        active[i] = true;
      }
      // Active novel inputs for growth
      for (let i = 35; i < 40; i++) {
        active[i] = true;
      }
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

  function renderAt(t) {
    if (!simData) return;
    const clampedStep = Math.max(0, Math.min(NUM_STEPS - 1, Math.floor(t)));
    currentStep = clampedStep;
    const sData = simData.steps[clampedStep];

    // 1. Draw the hero canvas, easing the bead sizes between steps.
    const st = segHeroState(simData, t);
    drawCanvas(st);
    heroCap.set(st.narration);

    // 2. Status note
    let actDetails = [];
    for (let s = 0; s < 2; s++) {
      const segAct = sData.actions[s] || [];
      const reinforced = segAct.filter((a) => a.action === 'reinforced').map((a) => `Syn ${a.synapseIdx}`);
      const punished = segAct.filter((a) => a.action === 'punished').map((a) => `Syn ${a.synapseIdx}`);
      const grown = segAct.filter((a) => a.action === 'grown').map((a) => `Syn ${a.synapseIdx}`);

      let statusMsg = sData.matching[s] ? 'matched' : 'not matched: no weight changes';
      let summary = `<strong>Segment ${s}</strong> (active connected = ${sData.activation[s]}, threshold ${learnThresh.get()}: <em>${statusMsg}</em>): `;
      const parts = [];
      if (reinforced.length) parts.push(`<span style="color:var(--c5,#5b8a3c)">reinforced [${reinforced.slice(0, 5).join(', ')}${reinforced.length > 5 ? '…' : ''}]</span>`);
      if (punished.length) parts.push(`<span style="color:var(--c2,#d95b43)">punished [${punished.slice(0, 5).join(', ')}${punished.length > 5 ? '…' : ''}]</span>`);
      if (grown.length) parts.push(`<span style="color:var(--c1,#0f7b7f)">grown [${grown.join(', ')}]</span>`);
      if (!parts.length) parts.push('no weight changes');
      actDetails.push(summary + parts.join('; '));
    }

    // The narration line already names the step, match state and depolarization,
    // so the details block only lists the per-segment synapse actions.
    statusBody.innerHTML = actDetails.join('<br>');

    // 3. Draw permanences plot over all steps with revealX
    // Plot exactly 6 representative synapses from Segment 0 with concise labels
    const representative = [
      { syn: 0, label: 'Syn 0 (active)', color: 'c1' },
      { syn: 2, label: 'Syn 2 (active)', color: 'c5' },
      { syn: 5, label: 'Syn 5 (active)', color: 'c4' },
      { syn: 10, label: 'Syn 10 (inactive)', color: 'c2' },
      { syn: 12, label: 'Syn 12 (inactive)', color: 'c3' },
      { syn: 15, label: 'Syn 15 (grown)', color: 'c6' },
    ];

    const series = representative.map((r) => {
      const xs = [];
      const ys = [];
      for (let t = 0; t < NUM_STEPS; t++) {
        xs.push(t);
        ys.push(simData.steps[t].seg0After.permanences[r.syn]);
      }
      return {
        x: xs,
        y: ys,
        label: r.label,
        color: r.color,
      };
    });

    permPlot.draw(series, {
      xRange: [0, NUM_STEPS - 1],
      yRange: [0, 1],
      revealX: clampedStep,
      hlines: [
        { y: connPerm.get(), label: `conn (${connPerm.get()})` },
      ],
    });
  }

  // Hero: two dendritic branches with synapses as beads whose size is the
  // eased permanence. Active presynaptic inputs light their beads; reinforced
  // beads grow (green), punished shrink (red), grown synapses sprout (teal).
  function drawCanvas(st) {
    const { canvas, ctx } = canvasFig;
    const w = canvas.width || 400;
    const h = canvas.height || 270;

    ctx.clearRect(0, 0, w, h);

    // Soma center at bottom
    const somaX = w * 0.45;
    const somaY = h - 40;
    const somaR = 24;

    ctx.save();

    // Draw soma
    ctx.beginPath();
    ctx.arc(somaX, somaY, somaR, 0, 2 * Math.PI);
    const depolVal = st.depol;
    ctx.fillStyle = depolVal >= 1.0 ? ui.color('c2') : ui.color('c1');
    ctx.globalAlpha = 0.2 + 0.8 * depolVal;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = ui.color('ink');
    ctx.lineWidth = 2;
    ctx.stroke();

    // Soma label inside circle
    ctx.fillStyle = ui.color('ink');
    ctx.font = 'bold 11px ui-sans-serif, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Soma', somaX, somaY);

    // Depolarization readout NEXT to the circle (no clipping!)
    ctx.textAlign = 'left';
    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.fillText(`Depol: ${(depolVal * 100).toFixed(0)}%`, somaX + somaR + 8, somaY - 6);
    ctx.font = '10px ui-sans-serif, sans-serif';
    ctx.fillStyle = depolVal >= 1.0 ? ui.color('c2') : ui.color('muted');
    ctx.fillText(depolVal >= 1.0 ? 'State: Predictive (depolarized)' : 'State: Subthreshold', somaX + somaR + 8, somaY + 8);

    // Draw 2 dendritic branches (Segment 0 left, Segment 1 right)
    const branches = [
      { segIdx: 0, startX: somaX - 10, startY: somaY - somaR, endX: w * 0.22, endY: 45 },
      { segIdx: 1, startX: somaX + 10, startY: somaY - somaR, endX: w * 0.78, endY: 45 },
    ];

    branches.forEach((b) => {
      const seg = st.segs[b.segIdx];
      const isMatching = seg.matching;
      const isActive = seg.active;

      // Trunk line
      ctx.beginPath();
      ctx.moveTo(b.startX, b.startY);
      ctx.quadraticCurveTo(
        b.startX + (b.endX - b.startX) * 0.35,
        b.startY - 50,
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
        `Segment ${b.segIdx} (${seg.activation} active)`,
        b.endX,
        b.endY - 14,
      );

      // Draw synapses along branch as beads sized by the eased permanence.
      const count = seg.syn.length;
      for (let i = 0; i < count; i++) {
        const tFrac = 0.12 + (i / count) * 0.82;
        // Point along quadratic curve
        const qx = (1 - tFrac) * (1 - tFrac) * b.startX + 2 * (1 - tFrac) * tFrac * (b.startX + (b.endX - b.startX) * 0.35) + tFrac * tFrac * b.endX;
        const qy = (1 - tFrac) * (1 - tFrac) * b.startY + 2 * (1 - tFrac) * tFrac * (b.startY - 50) + tFrac * tFrac * b.endY;

        const syn = seg.syn[i];
        const isConn = syn.perm >= connPerm.get();

        // Bead: radius = eased permanence (grown synapses sprout larger).
        ctx.beginPath();
        ctx.arc(qx, qy, syn.radius, 0, 2 * Math.PI);

        let synColor = ui.color('paper');
        if (syn.action === 'reinforced') synColor = ui.color('c5');
        else if (syn.action === 'punished') synColor = ui.color('c2');
        else if (syn.action === 'grown') synColor = ui.color('c1');
        else if (isConn) synColor = ui.color('c3');

        ctx.fillStyle = synColor;
        ctx.fill();
        ctx.strokeStyle = isConn ? ui.color('ink') : ui.color('muted');
        ctx.lineWidth = isConn ? 1.5 : 0.8;
        ctx.stroke();

        // Active presynaptic input: a short pulse up the branch to this bead.
        if (syn.active) {
          ctx.strokeStyle = ui.color('c5');
          ctx.globalAlpha = 0.5;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(qx, qy + 10);
          ctx.lineTo(qx, qy + syn.radius + 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    });

    ctx.restore();
  }

  // --- Animation / Player (direct integration per AMENDMENT 2) ---
  // integer:false so onFrame gets the fractional time and the hero eases the
  // bead sizes between steps; the plot still reveals at the integer step.
  const player = ui.player(figures, {
    duration: NUM_STEPS - 1,
    unit: 'step',
    playSeconds: 6,
    loop: true,
    integer: false,
    onFrame: (t) => {
      renderAt(t);
    },
  });

  function rebuild() {
    runSimulation();
    renderAt(currentStep);
  }

  [permInc, permDec, initPerm, connPerm, actThresh, learnThresh, maxNew, contextOverlap, seedCtrl].forEach((c) => {
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
    <p>Segment learning in CoNeX is implemented by <code>DendriticSegment.learn_on_segments</code> (active dendrites)
    and <code>SegmentStorage.adapt_segment, used by TemporalMemory</code> (sequence memory). Both enforce Hebbian structural plasticity:
    synapses that successfully predicted activation are <strong>reinforced</strong> (+<code>permanence_increment</code>),
    while co-allocated synapses from inactive presynaptic cells are <strong>punished</strong> (-<code>permanence_decrement</code>).
    Segments with room grow new synapses to active inputs with <code>initial_permanence</code> (using <code>SegmentStorage.grow_synapses</code>).</p>
    <p>When a segment causes a false prediction (the cell was predicted to fire but feedforward drive never arrived),
    <code>SegmentStorage.punish_segments</code> (used by <code>TemporalMemory</code>) decrements the active synapses that produced the false alarm,
    pruning non-predictive connections.</p>
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
    and the CoNeX defaults in <code>DendriticSegmentConfig</code> (<code>active_dendrites.py:35</code>) and <code>TemporalMemoryConfig</code> (<code>sequence_memory.py:40</code>).</p>
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
