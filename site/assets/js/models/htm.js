// HTM models mirroring conex/behaviors/neurons/numenta/{sdr,spatial_pooler,
// sequence_memory}.py. Pure math, no DOM. Parameter names and defaults match
// the Python classes.

// --- SDR -------------------------------------------------------------------

export class SDR {
  // indices: iterable of active bit positions (duplicates removed, sorted).
  constructor(size, indices = []) {
    if (!Number.isInteger(size) || size < 0) {
      throw new Error(`size must be a non-negative integer, got ${size}`);
    }
    this.size = size;
    this._indices = Array.from(new Set(indices)).sort((a, b) => a - b);
  }

  get indices() {
    return this._indices;
  }

  get n_active() {
    return this._indices.length;
  }

  get sparsity() {
    return this.size > 0 ? this.n_active / this.size : 0.0;
  }

  toDense() {
    const d = new Array(this.size).fill(0);
    for (const i of this._indices) d[i] = 1;
    return d;
  }

  // Number of shared active bits (set intersection size).
  overlap(other) {
    if (this.n_active === 0 || other.n_active === 0) return 0;
    const b = new Set(other._indices);
    let n = 0;
    for (const i of this._indices) if (b.has(i)) n++;
    return n;
  }

  // overlap / min(n_active) — the standard HTM similarity measure.
  overlapScore(other) {
    if (this.n_active === 0 || other.n_active === 0) return 0.0;
    return this.overlap(other) / Math.min(this.n_active, other.n_active);
  }

  jaccard(other) {
    const overlap = this.overlap(other);
    const unionSize = this.n_active + other.n_active - overlap;
    if (unionSize === 0) return 0.0;
    return overlap / unionSize;
  }

  union(other) {
    return new SDR(this.size, [...this._indices, ...other._indices]);
  }

  intersection(other) {
    if (this.n_active === 0 || other.n_active === 0) return new SDR(this.size);
    const b = new Set(other._indices);
    return new SDR(this.size, this._indices.filter((i) => b.has(i)));
  }

  difference(other) {
    if (this.n_active === 0) return new SDR(this.size);
    if (other.n_active === 0) return new SDR(this.size, this._indices);
    const b = new Set(other._indices);
    return new SDR(this.size, this._indices.filter((i) => !b.has(i)));
  }

  matchThreshold(other, theta) {
    return this.overlap(other) >= theta;
  }

  equals(other) {
    if (!(other instanceof SDR)) return false;
    if (this.size !== other.size || this.n_active !== other.n_active) return false;
    return this.overlap(other) === this.n_active;
  }
}

// SDRConfig.n_active: max(1, int(size * sparsity)).
export function sdrConfigNActive(size, sparsity) {
  return Math.max(1, Math.trunc(size * sparsity));
}

// --- False-match probability ----------------------------------------------

// log-gamma via Lanczos; used to evaluate the hypergeometric tail in log-space
// so tiny probabilities (e.g. 2.5e-26) do not underflow.
function lgamma(x) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function logComb(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

// P(B >= theta) for a random w-bit SDR overlapping a fixed w_x-bit SDR.
// Hypergeometric tail: sum_b C(w_x,b) C(n-w_x, w-b) / C(n, w)
// (Ahmad and Hawkins 2016, eq. 3).
export function falseMatchProbability(n, wx, w, theta) {
  if (n < 1 || wx < 0 || wx > n || w < 0 || w > n) {
    throw new Error(`need n>=1 and 0<=w,wx<=n, got n=${n}, wx=${wx}, w=${w}`);
  }
  if (theta < 0) throw new Error(`theta must be non-negative, got ${theta}`);
  const logDenom = logComb(n, w);
  let sum = 0;
  for (let b = theta; b <= Math.min(w, wx); b++) {
    const term = logComb(wx, b) + logComb(n - wx, w - b) - logDenom;
    sum += Math.exp(term);
  }
  return sum;
}

// Expected overlap of two independent uniform w-bit SDRs: w^2 / n.
export function expectedOverlap(n, w) {
  return (w * w) / n;
}

// --- SDREncoder (scalar) ---------------------------------------------------

// Mirrors SDREncoder.encode_scalar: a contiguous block of n_active bits whose
// center tracks the normalized value, wrapping around at the end.
export function encodeScalar(value, { size, n_active, min_val, max_val }) {
  const normalized = Math.max(0.0, Math.min(1.0, (value - min_val) / (max_val - min_val)));
  const center = Math.trunc(normalized * (size - n_active));
  const indices = [];
  for (let i = 0; i < n_active; i++) indices.push((center + i) % size);
  return new SDR(size, indices);
}

// --- SDRClassifier ---------------------------------------------------------

// Mirrors SDRClassifier: per-class weight vectors over the SDR bits.
// learn() adds learning_rate to the active bits of the labelled class;
// infer() scores each class as (weights . dense) / (count + 1e-6).
export class SDRClassifier {
  constructor({ n_classes, sdr_size, learning_rate = 0.01 }) {
    this.n_classes = n_classes;
    this.sdr_size = sdr_size;
    this.learning_rate = learning_rate;
    this.weights = Array.from({ length: n_classes }, () => new Float64Array(sdr_size));
    this.counts = new Float64Array(n_classes);
  }

  learn(sdr, label) {
    if (label < 0 || label >= this.n_classes) {
      throw new Error(`label ${label} out of range [0, ${this.n_classes})`);
    }
    for (const i of sdr.indices) this.weights[label][i] += this.learning_rate;
    this.counts[label] += 1;
  }

  // Returns [predictedClass, scores[]].
  infer(sdr) {
    const dense = sdr.toDense();
    const scores = new Array(this.n_classes).fill(0);
    for (let c = 0; c < this.n_classes; c++) {
      let s = 0;
      for (let i = 0; i < this.sdr_size; i++) s += this.weights[c][i] * dense[i];
      scores[c] = s / (this.counts[c] + 1e-6);
    }
    let predicted = 0;
    for (let c = 1; c < this.n_classes; c++) if (scores[c] > scores[predicted]) predicted = c;
    return [predicted, scores];
  }
}

// --- UnionSDR --------------------------------------------------------------

export class UnionSDR {
  constructor(size) {
    this.size = size;
    this._members = [];
    this._indices = new Set();
  }

  add(sdr) {
    this._members.push(sdr);
    for (const i of sdr.indices) this._indices.add(i);
  }

  remove(sdr) {
    this._members = this._members.filter((m) => !m.equals(sdr));
    this._recompute();
  }

  _recompute() {
    this._indices = new Set();
    for (const m of this._members) for (const i of m.indices) this._indices.add(i);
  }

  contains(sdr, threshold = 0.5) {
    const u = new SDR(this.size, this._indices);
    return u.overlapScore(sdr) >= threshold;
  }

  toSDR() {
    return new SDR(this.size, this._indices);
  }

  get n_active() {
    return this._indices.size;
  }

  get sparsity() {
    return this.size > 0 ? this.n_active / this.size : 0.0;
  }

  get n_members() {
    return this._members.length;
  }
}

// --- SpatialPooler ---------------------------------------------------------

// SpatialPoolerConfig defaults (conex/behaviors/neurons/numenta/spatial_pooler.py).
export const SpatialPoolerDefaults = {
  input_size: 784,
  n_columns: 2048,
  potential_radius: 16,
  potential_pct: 0.5,
  global_inhibition: true,
  local_area_density: 0.02,
  stimulus_threshold: 0,
  syn_perm_inactive_dec: 0.008,
  syn_perm_active_inc: 0.05,
  syn_perm_connected: 0.1,
  min_pct_overlap_duty_cycles: 0.001,
  duty_cycle_period: 1000,
  boost_strength: 0.0,
  seed: -1,
};

// Mirrors SpatialPooler.compute: overlap -> boost -> inhibit (top-k) ->
// learn (permanence inc/dec) -> duty cycles -> boost factors.
export class SpatialPooler {
  constructor(config = {}) {
    this.cfg = { ...SpatialPoolerDefaults, ...config };
    this.n_columns = this.cfg.n_columns;
    this.n_active = Math.max(1, Math.trunc(this.cfg.n_columns * this.cfg.local_area_density));
    this.n_potential = Math.max(1, Math.trunc(this.cfg.input_size * this.cfg.potential_pct));
    // state (set by the caller, mirroring the tensors CoNeX allocates)
    this.potential_inputs = null; // [n_columns][n_potential] input indices
    this.permanences = null; // [n_columns][n_potential]
    this.boost_factors = null; // [n_columns]
    this.overlap_duty = null; // [n_columns]
    this.active_duty = null; // [n_columns]
    this.active_columns = null; // [n_columns] bool
    this.overlap = null; // [n_columns]
    this.iteration = 0;
  }

  // input: array of 0/1 of length input_size. Returns active column indices.
  compute(input, learn = true) {
    const { n_columns, n_potential, connected_perm = this.cfg.syn_perm_connected } = this;
    this.iteration += 1;

    // Phase 1: overlap = count of connected synapses with active inputs.
    const overlap = new Array(n_columns).fill(0);
    for (let c = 0; c < n_columns; c++) {
      let o = 0;
      for (let p = 0; p < n_potential; p++) {
        const perm = this.permanences[c][p];
        if (perm >= connected_perm && input[this.potential_inputs[c][p]] > 0) o++;
      }
      if (o < this.cfg.stimulus_threshold) o = 0;
      overlap[c] = o;
    }
    this.overlap = overlap;

    // Phase 2: boosting.
    const boosted = overlap.map((o, c) => o * this.boost_factors[c]);

    // Phase 3: inhibition (global top-k; local branch is identical in CoNeX).
    const active = new Array(n_columns).fill(false);
    if (boosted.reduce((a, b) => a + b, 0) > 0) {
      const order = boosted
        .map((o, c) => [o, c])
        .sort((a, b) => b[0] - a[0] || a[1] - b[1])
        .slice(0, this.n_active);
      for (const [, c] of order) active[c] = true;
    }
    this.active_columns = active;

    // Phase 4: learning.
    if (learn) {
      this._learn(input);
      this._updateDutyCycles();
      this._updateBoostFactors();
    }
    return active.map((a, c) => (a ? c : -1)).filter((c) => c >= 0);
  }

  _learn(input) {
    const { n_potential } = this;
    for (let c = 0; c < this.active_columns.length; c++) {
      if (!this.active_columns[c]) continue;
      for (let p = 0; p < n_potential; p++) {
        const activeInput = input[this.potential_inputs[c][p]] > 0;
        this.permanences[c][p] += activeInput
          ? this.cfg.syn_perm_active_inc
          : -this.cfg.syn_perm_inactive_dec;
      }
    }
    for (let c = 0; c < this.permanences.length; c++) {
      for (let p = 0; p < n_potential; p++) {
        this.permanences[c][p] = Math.max(0, Math.min(1, this.permanences[c][p]));
      }
    }
  }

  _updateDutyCycles() {
    const period = Math.min(this.iteration, this.cfg.duty_cycle_period);
    const alpha = 1.0 / period;
    for (let c = 0; c < this.overlap_duty.length; c++) {
      const overlapActive = this.overlap[c] > 0 ? 1 : 0;
      this.overlap_duty[c] = (1 - alpha) * this.overlap_duty[c] + alpha * overlapActive;
      this.active_duty[c] =
        (1 - alpha) * this.active_duty[c] + alpha * (this.active_columns[c] ? 1 : 0);
    }
  }

  _updateBoostFactors() {
    if (this.cfg.boost_strength <= 0) return;
    const targetDuty = this.cfg.local_area_density;
    for (let c = 0; c < this.boost_factors.length; c++) {
      this.boost_factors[c] = Math.exp(
        this.cfg.boost_strength * (targetDuty - this.active_duty[c])
      );
    }
  }
}

// --- TemporalMemory --------------------------------------------------------

// TemporalMemoryConfig defaults (conex/behaviors/neurons/numenta/sequence_memory.py).
export const TemporalMemoryDefaults = {
  n_columns: 2048,
  cells_per_column: 32,
  activation_threshold: 13,
  learning_threshold: 10,
  initial_permanence: 0.21,
  connected_permanence: 0.5,
  permanence_increment: 0.1,
  permanence_decrement: 0.1,
  predicted_decrement: 0.0,
  max_segments_per_cell: 255,
  max_synapses_per_segment: 255,
  max_new_synapses: 20,
};

// Mirrors SegmentStorage: per-cell list of segments, each a list of
// {target, permanence} synapses.
export class SegmentStorage {
  constructor(n_cells, max_segments, max_synapses, max_new_synapses = 20) {
    this.n_cells = n_cells;
    this.max_segments = max_segments;
    this.max_synapses = max_synapses;
    this.max_new_synapses = max_new_synapses;
    this.segments = Array.from({ length: n_cells }, () => []);
  }

  countSegments(cellIdx) {
    return this.segments[cellIdx].length;
  }

  // A segment is active when >= threshold connected synapses point at active cells.
  hasActiveSegment(cellIdx, activeCells, threshold, connectedPerm) {
    for (const seg of this.segments[cellIdx]) {
      let n = 0;
      for (const s of seg) if (s.permanence >= connectedPerm && activeCells[s.target]) n++;
      if (n >= threshold) return true;
    }
    return false;
  }

  // Best segment score = max over segments of (synapses to active cells).
  getBestSegmentScore(cellIdx, activeCells) {
    let best = 0;
    for (const seg of this.segments[cellIdx]) {
      let score = 0;
      for (const s of seg) if (activeCells[s.target]) score++;
      if (score > best) best = score;
    }
    return best;
  }

  // Active segment (>= activation_threshold) else best matching segment
  // (>= learning_threshold). Returns segment index or -1.
  getActiveOrMatchingSegment(cellIdx, activeCells, activationThreshold, learningThreshold, connectedPerm) {
    let bestSegment = -1;
    let bestScore = 0;
    for (let segIdx = 0; segIdx < this.segments[cellIdx].length; segIdx++) {
      const seg = this.segments[cellIdx][segIdx];
      let nConnectedActive = 0;
      let nActive = 0;
      for (const s of seg) {
        if (activeCells[s.target]) nActive++;
        if (s.permanence >= connectedPerm && activeCells[s.target]) nConnectedActive++;
      }
      if (nConnectedActive >= activationThreshold) return segIdx;
      if (nActive >= learningThreshold && nActive > bestScore) {
        bestScore = nActive;
        bestSegment = segIdx;
      }
    }
    return bestSegment;
  }

  adaptSegment(cellIdx, segmentIdx, activeCells, permInc, permDec) {
    const seg = this.segments[cellIdx][segmentIdx];
    if (!seg) return;
    for (const s of seg) {
      s.permanence += activeCells[s.target] ? permInc : -permDec;
      s.permanence = Math.max(0, Math.min(1, s.permanence));
    }
  }

  // Grow up to max_new synapses to winner cells not already targeted.
  // newTargets: array of winner cell indices to add (the caller selects the
  // subset; CoNeX samples with torch.randperm, which is order-only when the
  // whole candidate set is taken).
  growSynapses(cellIdx, segmentIdx, newTargets, initialPerm) {
    const seg = this.segments[cellIdx][segmentIdx];
    if (!seg) return;
    const current = new Set(seg.map((s) => s.target));
    if (current.size >= this.max_synapses) return;
    const candidates = newTargets.filter((t) => !current.has(t));
    if (candidates.length === 0) return;
    const nNew = Math.min(candidates.length, this.max_new_synapses, this.max_synapses - current.size);
    for (let i = 0; i < nNew; i++) seg.push({ target: candidates[i], permanence: initialPerm });
  }

  // New segment targeting up to max_new of the given winner cells.
  createSegment(cellIdx, winnerCells, initialPerm) {
    if (this.segments[cellIdx].length >= this.max_segments) return -1;
    const n = Math.min(winnerCells.length, this.max_new_synapses);
    if (n === 0) return -1;
    const seg = [];
    for (let i = 0; i < n; i++) seg.push({ target: winnerCells[i], permanence: initialPerm });
    this.segments[cellIdx].push(seg);
    return this.segments[cellIdx].length - 1;
  }

  punishSegments(cellIdx, activeCells, decrement, threshold, connectedPerm) {
    for (const seg of this.segments[cellIdx]) {
      let n = 0;
      for (const s of seg) if (s.permanence >= connectedPerm && activeCells[s.target]) n++;
      if (n >= threshold) {
        for (const s of seg) {
          if (activeCells[s.target]) s.permanence -= decrement;
          s.permanence = Math.max(0, Math.min(1, s.permanence));
        }
      }
    }
  }
}

// Mirrors TemporalMemory.compute: activate cells (predicted-only or burst),
// compute predictions, then learn (adapt/grow/create segments).
export class TemporalMemory {
  constructor(config = {}) {
    this.cfg = { ...TemporalMemoryDefaults, ...config };
    this.n_columns = this.cfg.n_columns;
    this.cells_per_column = this.cfg.cells_per_column;
    this.n_cells = this.n_columns * this.cells_per_column;
    this.segments = new SegmentStorage(
      this.n_cells,
      this.cfg.max_segments_per_cell,
      this.cfg.max_synapses_per_segment,
      this.cfg.max_new_synapses
    );
    this.active_cells = new Array(this.n_cells).fill(false);
    this.predictive_cells = new Array(this.n_cells).fill(false);
    this.winner_cells = new Array(this.n_cells).fill(false);
    this.active_columns = new Array(this.n_columns).fill(false);
    this.bursting_columns = new Array(this.n_columns).fill(false);
    this.prev_active_cells = new Array(this.n_cells).fill(false);
    this.prev_winner_cells = new Array(this.n_cells).fill(false);
    this.anomaly_score = 0.0;
    this.prediction_accuracy = 0.0;
  }

  // activeColumns: array of active column indices.
  compute(activeColumns, learn = true) {
    const { n_cells, n_columns, cells_per_column } = this;

    // Store previous state (used by learning and winner selection).
    this.prev_active_cells = this.active_cells.slice();
    this.prev_winner_cells = this.winner_cells.slice();

    this.active_columns = new Array(n_columns).fill(false);
    for (const c of activeColumns) this.active_columns[c] = true;

    // Phase 1: activate cells.
    this.active_cells = new Array(n_cells).fill(false);
    this.winner_cells = new Array(n_cells).fill(false);
    this.bursting_columns = new Array(n_columns).fill(false);
    for (const c of activeColumns) {
      const start = c * cells_per_column;
      const end = start + cells_per_column;
      let predictedInCol = false;
      for (let i = start; i < end; i++) if (this.predictive_cells[i]) predictedInCol = true;
      if (predictedInCol) {
        for (let i = start; i < end; i++) this.active_cells[i] = this.predictive_cells[i];
        const winner = this._bestMatchingCell(start, end, this.prev_active_cells);
        if (winner >= 0) this.winner_cells[winner] = true;
      } else {
        for (let i = start; i < end; i++) this.active_cells[i] = true;
        this.bursting_columns[c] = true;
        let winner = this._bestMatchingCell(start, end, this.prev_active_cells);
        if (winner < 0) winner = this._leastUsedCell(start, end);
        this.winner_cells[winner] = true;
      }
    }

    // Phase 2: predictions for the next timestep.
    this.predictive_cells = new Array(n_cells).fill(false);
    for (let i = 0; i < n_cells; i++) {
      if (
        this.segments.hasActiveSegment(
          i,
          this.active_cells,
          this.cfg.activation_threshold,
          this.cfg.connected_permanence
        )
      ) {
        this.predictive_cells[i] = true;
      }
    }

    // Phase 3: learning.
    if (learn) this._learn();

    // Anomaly = fraction of active columns that burst.
    const nActive = activeColumns.length;
    if (nActive === 0) {
      this.anomaly_score = 0.0;
    } else {
      const nBursting = this.bursting_columns.filter(Boolean).length;
      this.anomaly_score = nBursting / nActive;
      this.prediction_accuracy = 1.0 - this.anomaly_score;
    }
  }

  // Cell with the highest segment score against activeCells; lowest index
  // wins ties (CoNeX keeps the first strictly-better cell). -1 if none.
  _bestMatchingCell(start, end, activeCells) {
    let bestCell = -1;
    let bestScore = 0;
    for (let i = start; i < end; i++) {
      const score = this.segments.getBestSegmentScore(i, activeCells);
      if (score > bestScore) {
        bestScore = score;
        bestCell = i;
      }
    }
    return bestCell;
  }

  _leastUsedCell(start, end) {
    let leastUsed = start;
    let minSegments = Infinity;
    for (let i = start; i < end; i++) {
      const n = this.segments.countSegments(i);
      if (n < minSegments) {
        minSegments = n;
        leastUsed = i;
      }
    }
    return leastUsed;
  }

  _learn() {
    const {
      activation_threshold,
      learning_threshold,
      connected_permanence,
      permanence_increment,
      permanence_decrement,
      initial_permanence,
    } = this.cfg;
    const prevActive = this.prev_active_cells;
    const prevWinners = this.prev_winner_cells.map((w, i) => (w ? i : -1)).filter((i) => i >= 0);

    for (const cellIdx of this.winner_cells.map((w, i) => (w ? i : -1)).filter((i) => i >= 0)) {
      const segIdx = this.segments.getActiveOrMatchingSegment(
        cellIdx,
        prevActive,
        activation_threshold,
        learning_threshold,
        connected_permanence
      );
      if (segIdx >= 0) {
        this.segments.adaptSegment(
          cellIdx,
          segIdx,
          prevActive,
          permanence_increment,
          permanence_decrement
        );
        this.segments.growSynapses(cellIdx, segIdx, prevWinners, initial_permanence);
      } else {
        this.segments.createSegment(cellIdx, prevWinners, initial_permanence);
      }
    }

    if (this.cfg.predicted_decrement > 0) this._punishPredicted(prevActive);
  }

  _punishPredicted(prevActive) {
    const { activation_threshold, connected_permanence, predicted_decrement } = this.cfg;
    for (let c = 0; c < this.n_columns; c++) {
      if (this.active_columns[c]) continue;
      const start = c * this.cells_per_column;
      const end = start + this.cells_per_column;
      for (let i = start; i < end; i++) {
        if (this.predictive_cells[i]) {
          this.segments.punishSegments(
            i,
            prevActive,
            predicted_decrement,
            activation_threshold,
            connected_permanence
          );
        }
      }
    }
  }
}
