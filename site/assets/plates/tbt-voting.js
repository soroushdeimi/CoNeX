import { ColumnVoting, ConsensusNetwork } from '../js/models/tbt.js';

export const meta = {
  id: 'tbt-voting',
  title: 'Column voting and consensus',
  conex: ['ColumnVoting', 'ConsensusNetwork', 'SDROverlap'],
  source: 'conex/behaviors/network/numenta/voting.py',
};

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

  // --- state ---
  let memories = [];
  let localReps = [];
  let columns = [];
  let consensus = null;
  let stepCount = 0;
  const trace = []; // per step: top accumulator per column + consensus flag

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
    trace.length = 0;
  }
  const rebuild = () => { build(); render(); };
  [nObjects, repSize, sparsity, voteThresh, decay, minVotes, nColumns, trueObj, noise, convThresh, seed].forEach((c) => c.onInput(rebuild));

  // --- figures ---
  const votePlot = ui.plot(figures, {
    height: 240, xLabel: 'voting step', xUnit: '', yLabel: 'vote accumulator', yUnit: 'a.u.',
  });
  const status = ui.note(figures, '<em>press Step or Run</em>');

  function stepOnce() {
    for (let c = 0; c < columns.length; c++) {
      const votes = columns[c].castVote(localReps[c], memories);
      columns[c].receiveVotes(votes);
    }
    consensus.broadcastVotes(columns);
    for (const col of columns) col.forward();
    const reached = columns.map((col) => col.checkConsensus(columns.length));
    const netConsensus = consensus.checkNetworkConsensus(columns);
    consensus.forward();
    stepCount++;
    // record the top accumulator of column 0 and the network consensus
    const acc0 = Array.from(columns[0].voteAccumulator);
    const topIdx = acc0.indexOf(Math.max(...acc0));
    trace.push({ step: stepCount, top: acc0[topIdx], topIdx, reached: reached.every(Boolean), net: netConsensus });
    if (trace.length > 200) trace.shift();
  }

  function render() {
    const t = trace;
    votePlot.draw(
      [
        { x: t.map((p) => p.step), y: t.map((p) => p.top), label: 'top vote (col 0)', color: 'c1' },
      ],
      { xRange: [0, Math.max(1, stepCount)] },
    );
    const last = t[t.length - 1];
    if (last) {
      const objs = columns.map((c) => c.consensusObject);
      status.innerHTML = `step ${stepCount} &nbsp;|&nbsp; column consensus: ${
        objs.map((o) => (o < 0 ? '&ndash;' : 'obj ' + o)).join(', ')
      } &nbsp;|&nbsp; network consensus: ${last.net ? '<strong>yes</strong>' : 'no'}`;
    } else {
      status.innerHTML = '<em>press Step or Run</em>';
    }
  }

  ui.button(controls, { label: 'Step', onClick: () => { stepOnce(); render(); } });
  ui.button(controls, { label: 'Run 20 steps', onClick: () => { for (let i = 0; i < 20; i++) stepOnce(); render(); } });
  ui.button(controls, { label: 'Reset', onClick: () => { build(); render(); } });

  build();
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
  `);
  ui.equation(notes, `
    <p>vote<sub>o</sub> = |local &cap; memory<sub>o</sub>| / |local|, &nbsp; kept only if &ge; voting_threshold</p>
    <p>accumulator<sub>o</sub> &larr; accumulator<sub>o</sub> &middot; (1 &minus; decay_rate) + incoming<sub>o</sub></p>
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
from conex import ColumnVoting, ConsensusNetwork

voting = ColumnVoting(
    n_objects=${nObjects.get()},
    representation_size=${repSize.get()},
    sparsity=${sparsity.get()},
    voting_threshold=${voteThresh.get()},
    decay_rate=${decay.get()},
    min_votes_for_consensus=${minVotes.get()},
)
consensus = ConsensusNetwork(convergence_threshold=${convThresh.get()})

votes = voting.cast_vote(column, local_representation, object_memories)
voting.receive_votes(column, votes)
consensus.broadcast_votes(network, columns)
voting.check_consensus(column, n_columns=${nColumns.get()})
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
