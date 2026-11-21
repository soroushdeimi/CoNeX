// Plate: STDP rule family across synaptic connection topologies.
// Covers all 13 CoNeX learning classes:
// SimpleSTDP, SparseSTDP, One2OneSTDP, Conv2dSTDP, Local2dSTDP,
// SimpleiSTDP, One2OneiSTDP, SparseiSTDP,
// SimpleRSTDP, One2OneRSTDP, SparseRSTDP, Conv2dRSTDP, Local2dRSTDP.

import * as M from '../js/models/learning.js';

export const meta = {
  id: 'learn-stdp-family',
  title: 'STDP rule family',
  conex: [
    'SimpleSTDP', 'SparseSTDP', 'One2OneSTDP', 'Conv2dSTDP', 'Local2dSTDP',
    'SimpleiSTDP', 'One2OneiSTDP', 'SparseiSTDP',
    'SimpleRSTDP', 'One2OneRSTDP', 'SparseRSTDP', 'Conv2dRSTDP', 'Local2dRSTDP',
  ],
  source: 'conex/behaviors/synapses/learning.py',
};

// Generates reproducible Bernoulli spike trains from a seed for 16 pre and 4 post neurons over T steps.
function generateSpikeTrains(seed, steps = 40) {
  let s = (seed >>> 0) || 1;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const preSpikes16 = [];
  const postSpikes4 = [];
  for (let t = 0; t < steps; t++) {
    const preRow = [];
    for (let i = 0; i < 16; i++) {
      preRow.push(rand() < 0.28 ? 1 : 0);
    }
    preSpikes16.push(preRow);

    const postRow = [];
    for (let j = 0; j < 4; j++) {
      postRow.push(rand() < 0.32 ? 1 : 0);
    }
    postSpikes4.push(postRow);
  }
  return { preSpikes16, postSpikes4 };
}

// Build topological metadata and initial weights for each connection type
function buildTopology(connection, w0 = 0.5) {
  switch (connection) {
    case 'Sparse': {
      // 8 non-zero connections between 4 pre and 4 post neurons
      const src_idx = [0, 1, 0, 3, 0, 0, 2, 3];
      const dst_idx = [0, 0, 1, 1, 2, 3, 3, 3];
      return {
        is_sparse: true,
        src_idx,
        dst_idx,
        initial_weights: new Array(src_idx.length).fill(w0),
        rows: 4,
        cols: 4,
        toMatrix: (w) => {
          const mat = new Array(16).fill(0);
          for (let k = 0; k < src_idx.length; k++) {
            mat[src_idx[k] * 4 + dst_idx[k]] = w[k];
          }
          return mat;
        },
      };
    }
    case 'One2One': {
      // 4 one-to-one connections
      return {
        is_121: true,
        initial_weights: new Array(4).fill(w0),
        rows: 1,
        cols: 4,
        toMatrix: (w) => [...w],
      };
    }
    case 'Conv2d': {
      // 3x3 shared kernel sweeping 4x4 input to 2x2 output
      return {
        is_conv: true,
        kh: 3,
        kw: 3,
        stride: 1,
        padding: 0,
        srcH: 4,
        srcW: 4,
        dstH: 2,
        dstW: 2,
        initial_weights: new Array(9).fill(w0),
        rows: 3,
        cols: 3,
        toMatrix: (w) => [...w],
      };
    }
    case 'Local2d': {
      // 4 distinct 3x3 receptive fields for 4 destination neurons
      return {
        is_local: true,
        kh: 3,
        kw: 3,
        stride: 1,
        padding: 0,
        srcH: 4,
        srcW: 4,
        dstH: 2,
        dstW: 2,
        initial_weights: new Array(36).fill(w0),
        rows: 4,
        cols: 9,
        toMatrix: (w) => [...w],
      };
    }
    case 'Simple':
    default: {
      // 4x4 dense matrix
      return {
        initial_weights: new Array(16).fill(w0),
        rows: 4,
        cols: 4,
        toMatrix: (w) => [...w],
      };
    }
  }
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const state = {
    rule: 'STDP',
    connection: 'Simple',
    a_plus: 0.02,
    a_minus: 0.01,
    lr: 0.001,
    rho: 10.0,
    tau_s: 10.0,
    tau_c: 15.0,
    payoff: 1.0,
    tau_dopamine: 10.0,
    w_min: 0.0,
    w_max: 1.0,
    bound: 'no_bound',
    seed: 42,
    w0: 0.5,
    steps: 40,
  };

  let player = null;
  let cachedSimulation = null;
  let cachedTopology = null;
  let currentStep = state.steps;

  const heatmapFig = ui.heatmap(figures, { rows: 4, cols: 4, label: 'Weights heatmap' });
  const meanPlot = ui.plot(figures, {
    height: 200,
    xLabel: 'time step',
    xUnit: 'steps',
    yLabel: 'mean weight',
    yUnit: 'a.u.',
  });

  function resolveClassName() {
    let conn = state.connection;
    if (state.rule === 'iSTDP' && (conn === 'Conv2d' || conn === 'Local2d')) {
      conn = 'Simple'; // fallback for unsupported combination
    }
    if (state.rule === 'STDP') return `${conn}STDP`;
    if (state.rule === 'iSTDP') return `${conn}iSTDP`;
    return `${conn}RSTDP`;
  }

  function resolveDendriticInputClass(conn) {
    switch (conn) {
      case 'Sparse': return 'SparseDendriticInput';
      case 'One2One': return 'One2OneDendriticInput';
      case 'Conv2d': return 'Conv2dDendriticInput';
      case 'Local2d': return 'Local2dDendriticInput';
      case 'Simple':
      default: return 'SimpleDendriticInput';
    }
  }

  function recompute() {
    const className = resolveClassName();
    const topology = buildTopology(state.connection, state.w0);
    cachedTopology = topology;

    const { preSpikes16, postSpikes4 } = generateSpikeTrains(state.seed, state.steps);
    const pre_spikes = (topology.is_conv || topology.is_local)
      ? preSpikes16
      : preSpikes16.map((row) => row.slice(0, 4));
    const post_spikes = postSpikes4;

    const params = {
      a_plus: state.a_plus,
      a_minus: state.a_minus,
      lr: state.lr,
      rho: state.rho,
      tau_s: state.tau_s,
      tau_c: state.tau_c,
      tau_dopamine: state.tau_dopamine,
      payoff: state.payoff,
      w_min: state.w_min,
      w_max: state.w_max,
      positive_bound: state.bound,
      negative_bound: state.bound,
    };

    cachedSimulation = M.simulateLearningTrace({
      className,
      params,
      meta: topology,
      pre_spikes,
      post_spikes,
    });
  }

  function renderAt(step) {
    if (!cachedSimulation || !cachedTopology) recompute();
    const t = Math.max(0, Math.min(Math.floor(step), state.steps - 1));
    currentStep = t;

    const curWeights = cachedSimulation.history_weights[t];
    const heatValues = cachedTopology.toMatrix(curWeights);
    const minW = Math.min(0, ...heatValues);
    const maxW = Math.max(1, ...heatValues);
    heatmapFig.draw(heatValues, { min: minW, max: maxW });

    const xs = [];
    const ys = [];
    for (let i = 0; i < cachedSimulation.history_mean_weights.length; i++) {
      xs.push(i);
      ys.push(cachedSimulation.history_mean_weights[i]);
    }

    meanPlot.draw(
      [
        {
          x: xs,
          y: ys,
          label: 'mean weight',
          color: 'c1',
        },
      ],
      {
        xRange: [0, state.steps],
        revealX: t,
        hlines: [{ y: state.w0, label: 'w0' }],
      }
    );
  }

  function update() {
    recompute();
    renderAt(currentStep);
  }

  // --- Controls ---
  const secSetup = ui.section(controls, 'Model selection');
  ui.select(secSetup, {
    key: 'rule',
    label: 'Learning rule',
    value: state.rule,
    options: [
      { value: 'STDP', label: 'STDP (classic)' },
      { value: 'iSTDP', label: 'iSTDP (inhibitory)' },
      { value: 'RSTDP', label: 'RSTDP (reward-modulated)' },
    ],
    onInput: (v) => {
      state.rule = v;
      if (v === 'iSTDP' && (state.connection === 'Conv2d' || state.connection === 'Local2d')) {
        state.connection = 'Simple';
        connSelect.set('Simple');
      }
      update();
    },
  });

  const connSelect = ui.select(secSetup, {
    key: 'connection',
    label: 'Topology',
    value: state.connection,
    options: [
      { value: 'Simple', label: 'Simple (dense 4x4)' },
      { value: 'Sparse', label: 'Sparse (density 0.5)' },
      { value: 'One2One', label: 'One2One (1-to-1)' },
      { value: 'Conv2d', label: 'Conv2d (shared 3x3)' },
      { value: 'Local2d', label: 'Local2d (local 3x3)' },
    ],
    onInput: (v) => {
      if (state.rule === 'iSTDP' && (v === 'Conv2d' || v === 'Local2d')) {
        state.connection = 'Simple';
        connSelect.set('Simple');
      } else {
        state.connection = v;
      }
      update();
    },
  });

  const secParams = ui.section(controls, 'Learning parameters');
  ui.slider(secParams, {
    key: 'a_plus',
    label: 'a_plus (STDP)',
    unit: 'a.u.',
    min: 0,
    max: 0.05,
    step: 0.001,
    value: state.a_plus,
    onInput: (v) => { state.a_plus = v; update(); },
  });
  ui.slider(secParams, {
    key: 'a_minus',
    label: 'a_minus (STDP)',
    unit: 'a.u.',
    min: 0,
    max: 0.05,
    step: 0.001,
    value: state.a_minus,
    onInput: (v) => { state.a_minus = v; update(); },
  });
  ui.slider(secParams, {
    key: 'lr',
    label: 'lr (iSTDP)',
    unit: 'a.u.',
    min: 0.0001,
    max: 0.005,
    step: 0.0001,
    value: state.lr,
    onInput: (v) => { state.lr = v; update(); },
  });
  ui.slider(secParams, {
    key: 'rho',
    label: 'rho (iSTDP)',
    unit: 'Hz',
    min: 1,
    max: 30,
    step: 1,
    value: state.rho,
    onInput: (v) => { state.rho = v; update(); },
  });
  ui.slider(secParams, {
    key: 'tau_c',
    label: 'tau_c (RSTDP)',
    unit: 'ms',
    min: 1,
    max: 50,
    step: 1,
    value: state.tau_c,
    onInput: (v) => { state.tau_c = v; update(); },
  });
  ui.slider(secParams, {
    key: 'payoff',
    label: 'payoff (RSTDP)',
    unit: 'a.u.',
    min: 0,
    max: 2,
    step: 0.1,
    value: state.payoff,
    onInput: (v) => { state.payoff = v; update(); },
  });
  ui.slider(secParams, {
    key: 'tau_s',
    label: 'tau_s (trace decay)',
    unit: 'ms',
    min: 1,
    max: 50,
    step: 1,
    value: state.tau_s,
    onInput: (v) => { state.tau_s = v; update(); },
  });
  ui.slider(secParams, {
    key: 'w0',
    label: 'initial weight w0',
    unit: 'a.u.',
    min: 0,
    max: 1,
    step: 0.05,
    value: state.w0,
    onInput: (v) => { state.w0 = v; update(); },
  });
  ui.slider(secParams, {
    key: 'seed',
    label: 'random seed',
    unit: 'a.u.',
    min: 1,
    max: 100,
    step: 1,
    value: state.seed,
    onInput: (v) => { state.seed = v; update(); },
  });

  // Animation player
  if (typeof ui.player === 'function') {
    player = ui.player(figures, {
      duration: state.steps,
      unit: 'steps',
      playSeconds: 5,
      loop: true,
      onFrame: (t) => renderAt(t),
    });
  }

  ui.onTheme(update);
  update();

  // Notes
  ui.note(
    notes,
    `CoNeX provides 13 synaptic learning rule classes across three rule paradigms (STDP, iSTDP, RSTDP) ` +
    `and five connection topologies (Simple, Sparse, One2One, Conv2d, Local2d). Classic STDP adapts weights ` +
    `from pre- and post-synaptic spike traces to implement associative plasticity. Symmetric iSTDP (Vogels et al. 2011) ` +
    `adjusts inhibitory strengths toward a target firing set-point rho. Reward-modulated STDP (RSTDP, Frémaux & Gerstner 2016) ` +
    `accumulates eligibility traces that are multiplicatively gated by extracellular dopamine.`
  );

  ui.equation(
    notes,
    `STDP: <var>dw</var> = <var>a</var><sub>+</sub>·<var>pre_trace</var>·<var>post_spike</var> − <var>a</var><sub>−</sub>·<var>pre_spike</var>·<var>post_trace</var><br>` +
    `iSTDP: <var>dw</var> = <var>lr</var>·[<var>pre_spike</var>·(<var>&alpha;</var> − <var>post_trace</var>)·<var>sign</var> + <var>pre_trace</var>·<var>post_spike</var>]<br>` +
    `RSTDP: <var>c</var><sub>t+1</sub> = <var>c</var><sub>t</sub>(1 − 1/<var>&tau;</var><sub>c</sub>) + <var>dw</var><sub>stdp</sub>, &nbsp; <var>&Delta;w</var> = <var>c</var>·<var>dopamine</var>`
  );

  ui.note(
    notes,
    `Topological differences: In Simple (dense) and Sparse connections, individual synapses update independently. ` +
    `One2One matches pre and post neurons elementwise. Conv2d shares a single kernel across all receptive fields and averages ` +
    `the weight gradients across destination neurons (dw / (dstH·dstW)), while Local2d maintains dedicated receptive field ` +
    `weights for every destination neuron without spatial parameter sharing. Note: CoNeX does not define Conv2d or Local2d for iSTDP. ` +
    `ML analogy: STDP resembles contrastive Hebbian learning and local weight gradient descent; the analogy breaks because STDP operates ` +
    `on asynchronous spike event times rather than smooth continuous activation values.`
  );

  // CoNeX code snippet
  ui.code(notes, () => {
    const cls = resolveClassName();
    const dCls = resolveDendriticInputClass(state.connection);
    const isRSTDP = cls.includes('RSTDP');
    const isISTDP = cls.includes('iSTDP');

    let dendriteCall = `${dCls}(current_coef=1)`;
    if (state.connection === 'Conv2d' || state.connection === 'Local2d') {
      dendriteCall = `${dCls}(current_coef=1, stride=1, padding=0)`;
    }

    let learnCall = '';
    if (isISTDP) {
      learnCall = `${cls}(lr=${state.lr}, rho=${state.rho}, is_inhibitory=True)`;
    } else if (isRSTDP) {
      learnCall = `${cls}(a_plus=${state.a_plus}, a_minus=${state.a_minus}, tau_c=${state.tau_c}, w_min=${state.w_min}, w_max=${state.w_max})`;
    } else {
      learnCall = `${cls}(a_plus=${state.a_plus}, a_minus=${state.a_minus}, w_min=${state.w_min}, w_max=${state.w_max})`;
    }

    const imports = [cls, dCls, 'PreTrace', 'PostTrace', 'PreSpikeCatcher', 'PostSpikeCatcher'];
    if (isRSTDP) imports.push('Dopamine', 'Payoff');

    const lines = [
      `from conex import ${imports.join(', ')}`,
    ];
    if (isRSTDP) {
      lines.push(`Payoff(initial_payoff=${state.payoff})`);
      lines.push(`Dopamine(tau_dopamine=${state.tau_dopamine})`);
    }
    lines.push(learnCall);
    lines.push(dendriteCall);
    lines.push(`PreTrace(tau_s=${state.tau_s}); PostTrace(tau_s=${state.tau_s})`);
    return lines.join('\n');
  });

  return {
    destroy() {
      if (player && typeof player.pause === 'function') {
        player.pause();
      }
    },
  };
}
