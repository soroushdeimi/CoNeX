import { ColumnVoting, ConsensusNetwork } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-voting',
  title: 'Column voting and consensus',
  conex: ['ColumnVoting', 'ConsensusNetwork', 'SDROverlap'],
  source: 'conex/behaviors/network/numenta/voting.py',
};

// --- Pure mappings (model state -> drawn quantity / narration) -------------
// Exported so site/tests/tbt.test.mjs can test them by hand-computed values.

// A candidate bar's length (0..1) is its accumulator over the max accumulator.
export function barFraction(value, max) {
  if (max <= 0) return 0;
  return Math.max(0, Math.min(1, value / max));
}

// The object a column currently favours (highest accumulator).
export function topObject(acc) {
  let top = -1, topV = -Infinity;
  for (let o = 0; o < acc.length; o++) {
    if (acc[o] > topV) { topV = acc[o]; top = o; }
  }
  return top;
}

// Consensus: every column that has reached consensus agrees on one object.
export function consensusObject(objs) {
  const counts = {};
  for (const o of objs) if (o >= 0) counts[o] = (counts[o] || 0) + 1;
  let best = -1, bestCount = 0;
  for (const k in counts) if (counts[k] > bestCount) { bestCount = counts[k]; best = Number(k); }
  return bestCount > 0 ? best : -1;
}

// One-line narration driven by the model state.
export function votingNarration({ step, objs, net }) {
  const agree = consensusObject(objs);
  if (net) return `step ${step}: network consensus on object ${agree}`;
  if (agree >= 0) return `step ${step}: columns are converging on object ${agree}`;
  return `step ${step}: no consensus yet`;
}

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  // --- ColumnVoting parameters (CoNeX names and defaults) ---
  const sec = ui.section(controls, 'ColumnVoting');
  const nObjects = ui.number(sec, { key: 'nObjects', label: 'n_objects', unit: 'objects', value: 6 });
  const repSize = ui.number(sec, { key: 'repSize', label: 'representation_size', unit: 'bits', value: 64 });
  const sparsity = ui.slider(sec, {
    key: 'sparsity', label: 'sparsity', unit: 'fraction', min: 0.05, max: 0.5, step: 0.05, value: 0.25,
  });
  const voteThresh = ui.slider(sec, {
    key: 'votingThreshold', label: 'voting_threshold', unit: 'a.u.', min: 0.1, max: 1, step: 0.05, value: 0.5,
  });
  const decay = ui.slider(sec, {
    key: 'decayRate', label: 'decay_rate', unit: 'a.u.', min: 0, max: 0.5, step: 0.05, value: 0.1,
  });
  const minVotes = ui.slider(sec, {
    key: 'minVotes', label: 'min_votes_for_consensus', unit: 'votes', min: 1, max: 6, step: 1, value: 3,
  });
  const nColumns = ui.number(sec, { key: 'nColumns', label: 'n_columns', unit: 'columns', value: 3 });

  // --- scenario ---
  const scn = ui.section(controls, 'Scenario');
  const trueObj = ui.slider(scn, {
    key: 'trueObject', label: 'true object', unit: 'index', min: 0, max: 5, step: 1, value: 2,
  });
  const noise = ui.slider(scn, {
    key: 'noise', label: 'noise (bits flipped / column)', unit: 'bits', min: 0, max: 8, step: 1, value: 2,
  });
  const convThresh = ui.slider(scn, {
    key: 'convergenceThreshold', label: 'convergence_threshold', unit: 'fraction', min: 0.3, max: 1, step: 0.05, value: 0.8,
  });
  const seed = ui.number(scn, { key: 'seed', label: 'seed', unit: '', value: 7 });
  const nSteps = ui.slider(scn, {
    key: 'votingSteps', label: 'voting steps', unit: 'steps', min: 2, max: 30, step: 1, value: 10,
  });

  // --- state ---
  let memories = [];
  let localReps = [];
  let columns = [];
  let consensus = null;
  let stepCount = 0;
  let frames = []; // precomputed steps; the player indexes into these
  let currentStep = 0;

  function build() {
    const rng = mulberry(seed.get());
    const nObj = nObjects.get();
    const rep = repSize.get();
    const nActive = Math.max(1, Math.floor(rep * sparsity.get()));
    memories = [];
    for (let o = 0; o < nObj; o++) {
      const row = new Float64Array(rep);
      for (const i of pickDistinct(rng, rep, nActive)) row[i] = 1;
      memories.push(row);
    }
    const t = Math.min(trueObj.get(), nObj - 1);
    localReps = [];
    for (let c = 0; c < nColumns.get(); c++) {
      const rep2 = Float64Array.from(memories[t]);
      const on = [];
      const off = [];
      for (let i = 0; i < rep; i++) (rep2[i] > 0 ? on : off).push(i);
      const flips = Math.min(noise.get(), on.length, off.length);
      const r2 = mulberry(seed.get() + 100 + c);
      for (const i of pickDistinct(r2, on.length, flips)) rep2[on[i]] = 0;
      for (const i of pickDistinct(r2, off.length, flips)) rep2[off[i]] = 1;
      localReps.push(rep2);
    }
    columns = [];
    for (let c = 0; c < nColumns.get(); c++) {
      columns.push(
        new ColumnVoting({
          nObjects: nObj,
          representationSize: rep,
          sparsity: sparsity.get(),
          votingThreshold: voteThresh.get(),
          decayRate: decay.get(),
          minVotesForConsensus: minVotes.get(),
        }),
      );
    }
    consensus = new ConsensusNetwork({ convergenceThreshold: convThresh.get() });
    stepCount = 0;
    frames = [];
    lastVotes = [];
    // record the initial (pre-vote) state
    frames.push(snapshot());
    // Run the voting for the requested number of steps, recording a frame per
    // step (stop early if the network reaches consensus). The player only
    // indexes into these, so no per-frame recomputation.
    for (let i = 0; i < nSteps.get(); i++) {
      stepOnce();
      frames.push(snapshot());
      if (frames[frames.length - 1].net) break;
    }
    currentStep = frames.length - 1;
  }

  // Capture the full state of one voting step: each column's candidate set
  // (its votes) and accumulator, plus the consensus verdicts.
  let lastVotes = [];
  function snapshot() {
    return {
      step: stepCount,
      votes: lastVotes,
      acc: columns.map((col) => Array.from(col.voteAccumulator)),
      objs: columns.map((col) => col.consensusObject),
      net: consensus.networkConsensus,
    };
  }

  const rebuild = () => { build(); player.setDuration(frames.length - 1); render(); };
  [nObjects, repSize, sparsity, voteThresh, decay, minVotes, nColumns, trueObj, noise, convThresh, seed, nSteps].forEach((c) => c.onInput(rebuild));

  // --- figures: hero on top, plot below ---
  const hero = ui.hero(figures, { height: 240 });
  const votePlot = ui.plot(figures, {
    height: 240, xLabel: 'voting step', xUnit: 'index', yLabel: 'vote accumulator', yUnit: 'a.u.',
  });

  function stepOnce() {
    lastVotes = [];
    for (let c = 0; c < columns.length; c++) {
      const votes = columns[c].castVote(localReps[c], memories);
      columns[c].receiveVotes(votes);
      lastVotes.push(Array.from(votes));
    }
    consensus.broadcastVotes(columns);
    for (const col of columns) col.forward();
    const reached = columns.map((col) => col.checkConsensus(columns.length));
    const netConsensus = consensus.checkNetworkConsensus(columns);
    consensus.forward();
    stepCount++;
    void reached;
    void netConsensus;
  }

  // Draw each column as a card with one horizontal bar per candidate object.
  // Bar lengths ease between step k and k+1; vote particles travel between
  // adjacent columns; all cards glow the same colour at network consensus.
  function drawHero(f, fNext, frac) {
    const { ctx, width, height } = hero;
    ctx.clearRect(0, 0, width, height);
    const ink = ui.color('ink');
    const muted = ui.color('muted');
    const c1 = ui.color('c1');
    const c2 = ui.color('c2');
    const c5 = ui.color('c5');
    const n = columns.length;
    const nObj = nObjects.get();
    const boxW = Math.min(170, (width - 20) / n - 14);
    const boxH = height - 30;
    const topY = 26;
    const barH = Math.min(16, (boxH - 10) / nObj - 2);
    const xs = [];
    const net = f.net;
    const e = ui.ease(frac); // ease-in-out between step k and k+1
    for (let c = 0; c < n; c++) {
      const x = 10 + c * ((width - 20) / n) + ((width - 20) / n - boxW) / 2;
      xs.push(x);
      // card: glows the consensus colour when the network agrees
      ctx.strokeStyle = net ? c5 : muted;
      ctx.lineWidth = net ? 2.5 : 1;
      ctx.strokeRect(x, 10, boxW, boxH);
      ctx.lineWidth = 1;
      ctx.fillStyle = ink;
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText('col ' + c, x + 4, 20);
      // eased accumulator between this step and the next
      const acc = f.acc[c];
      const accN = fNext ? fNext.acc[c] : acc;
      let maxV = 1e-6;
      for (let o = 0; o < nObj; o++) maxV = Math.max(maxV, acc[o], accN[o]);
      let top = -1, topV = -1;
      for (let o = 0; o < nObj; o++) {
        const v = acc[o] + (accN[o] - acc[o]) * e;
        if (v > topV) { topV = v; top = o; }
      }
      for (let o = 0; o < nObj; o++) {
        const v = acc[o] + (accN[o] - acc[o]) * e;
        const y = topY + o * (barH + 2);
        const isTop = o === top;
        ctx.fillStyle = isTop ? c2 : muted;
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(String(o), x + 4, y + barH - 3);
        const bw = barFraction(v, maxV) * (boxW - 24);
        ctx.fillStyle = isTop ? c2 : c1;
        ctx.globalAlpha = v > 0 ? 0.85 : 0.15;
        ctx.fillRect(x + 16, y, Math.max(0, bw), barH - 3);
        ctx.globalAlpha = 1;
      }
    }
    // vote particles travel between adjacent columns each step (decorative)
    if (frac > 0 && frac < 1) {
      for (let c = 0; c < n - 1; c++) {
        const y = 10 + boxH / 2;
        const x0 = xs[c] + boxW;
        const x1 = xs[c + 1];
        const nP = 3;
        for (let p = 0; p < nP; p++) {
          const pf = Math.max(0, Math.min(1, frac * 1.3 - p * 0.15));
          const px = x0 + (x1 - x0) * pf;
          ctx.globalAlpha = 0.8 * (1 - pf * 0.4);
          ctx.fillStyle = c1;
          ctx.beginPath(); ctx.arc(px, y, 2.5, 0, 2 * Math.PI); ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }
    hero.narration(votingNarration({ step: f.step, objs: f.objs, net: f.net }));
  }

  function render() {
    const f = frames[currentStep];
    if (!f) return;
    const i = Math.max(0, Math.min(frames.length - 1, Math.floor(currentStep)));
    const fNext = frames[i + 1] || frames[i];
    const frac = currentStep - i;
    drawHero(f, fNext, frac);
    // one series per column: its top accumulator at each step, revealed up to now
    const series = columns.map((col, c) => ({
      x: frames.map((p) => p.step),
      y: frames.map((p) => Math.max(...p.acc[c])),
      label: 'col ' + c + ' top',
      color: ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'][c % 6],
    }));
    votePlot.draw(series, { xRange: [0, Math.max(1, frames.length - 1)], revealX: f.step });
  }

  // The player time is the step index (0..nSteps); the fractional part eases
  // the bars between step k and k+1.
  function onFrame(t) {
    currentStep = Math.max(0, Math.min(frames.length - 1, t));
    render();
  }

  // The hero canvas must draw inside onResize (called after layout and on
  // every resize); drawing once at mount with a 0 width would leave it empty.
  hero.onResize(render);

  // Create the player before build() so rebuild() can resize it; the player
  // calls onFrame synchronously on creation, which renders the current frame.
  const player = ui.player(figures, {
    duration: nSteps.get(),
    unit: 'steps',
    playSeconds: 6,
    loop: true,
    onFrame,
  });
  build();
  player.setDuration(frames.length - 1);
  // Start at the final (converged) frame so the plot is populated on load;
  // the player then replays the voting from step 0.
  render();
  ui.onTheme(render);

  // --- notes ---
  ui.note(notes, `
    <p>Each cortical column builds its own model of objects. Because a column only
    sees a partial, noisy view, it keeps several <em>object hypotheses</em> and
    casts a vote for each one whose stored SDR overlaps its local representation
    by at least <code>voting_threshold</code>. Columns share votes laterally
    (averaged and broadcast) and accumulate them with a decay, so old votes fade.
    Consensus is reached when one object's votes lead the runner-up by a margin
    and exceed <code>min_votes_for_consensus</code>; the network agrees when a
    sufficient fraction of columns reach the same object.</p>
    <p>SDR voting is robust to noise: a partial overlap still counts, which is why
    a few flipped bits (the noise slider) do not stop the columns from converging
    on the true object.</p>
    <p>The hero shows each column as a card with one bar per candidate object; the
    bar's length is that object's vote accumulator, eased between voting steps.
    Vote particles travel between adjacent columns as the network broadcasts. When
    the network reaches consensus, every card glows the same colour. The plot below
    reveals each column's top accumulator over the steps. Every vote, bar, and
    accumulator shown is the real value from that step.</p>
  `);
  ui.equation(notes, `
    <p>vote<sub>o</sub> = |local &cap; memory<sub>o</sub>| / |local|, &nbsp; kept only if &ge; voting_threshold</p>
    <p>accumulator<sub>o</sub> &larr; accumulator<sub>o</sub> &middot; (1 &minus; decay_rate) + incoming<sub>o</sub> &nbsp;(in receive_votes)</p>
    <p>accumulator<sub>o</sub> &larr; accumulator<sub>o</sub> &middot; (1 &minus; decay_rate) &nbsp;(again in forward)</p>
    <p>so per step the accumulator is multiplied by (1 &minus; decay_rate)<sup>2</sup></p>
    <p>consensus: top &ge; min_votes and top / second &ge; 1.5</p>
  `);
  ui.note(notes, `
    <p><strong>ML analogy:</strong> this is a distributed, noisy version of
    majority voting / ensemble agreement across weak classifiers. The analogy
    breaks because the "votes" are high-dimensional sparse patterns compared by
    bit overlap rather than scalar scores, and the accumulator decays over time
    instead of being a fixed tally.</p>
  `);

  ui.code(notes, () => `
import torch
from pymonntorch import Network, NeuronGroup
from conex import ColumnVoting, ConsensusNetwork, TimeResolution

voting = ColumnVoting(
    n_objects=${nObjects.get()},
    representation_size=${repSize.get()},
    sparsity=${fmt(sparsity.get())},
    voting_threshold=${fmt(voteThresh.get())},
    decay_rate=${fmt(decay.get())},
    min_votes_for_consensus=${minVotes.get()},
)
consensus = ConsensusNetwork(convergence_threshold=${fmt(convThresh.get())})

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
columns = [
    NeuronGroup(net=net, size=${repSize.get()}, behavior={100: voting}, tag=f"col{c}")
    for c in range(${nColumns.get()})
]
net.initialize()
consensus.initialize(net)

# object memories: fixed random SDRs; local rep = object 0 with a few bits flipped
torch.manual_seed(${seed.get()})
n_active = max(1, int(${repSize.get()} * ${fmt(sparsity.get())}))
object_memories = torch.zeros(${nObjects.get()}, ${repSize.get()})
for o in range(${nObjects.get()}):
    object_memories[o, torch.randperm(${repSize.get()})[:n_active]] = 1.0
local_representation = object_memories[0].clone()
local_representation[:2] = 0.0
local_representation[-2:] = 1.0

votes = voting.cast_vote(columns[0], local_representation, object_memories)
voting.receive_votes(columns[0], votes)
consensus.broadcast_votes(net, columns)
voting.check_consensus(columns[0], n_columns=${nColumns.get()})
`);

  return { destroy() {} };
}

function mulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickDistinct(rng, n, k) {
  const pool = Array.from({ length: n }, (_, i) => i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, Math.min(k, n));
}

// At most 4 significant digits for numbers shown in text and code snippets.
function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  return Number(v.toPrecision(4)).toString();
}
