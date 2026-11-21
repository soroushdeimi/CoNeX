// Pure math for the Thousand Brains & predictive coding area.
// Mirrors conex/behaviors/neurons/numenta/{grid_cells,active_dendrites,predictive_coding}.py,
// conex/behaviors/network/numenta/voting.py and
// conex/nn/structure/numenta/predictive_hierarchy.py.
// No DOM, no randomness: every input is passed in.

const TWO_PI = 2 * Math.PI;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// ---------------------------------------------------------------------------
// GridCellModule  (conex/behaviors/neurons/numenta/grid_cells.py)
// ---------------------------------------------------------------------------

export class GridCellModule {
  constructor({
    nModules = 4,
    cellsPerModule = 40,
    scaleRange = [0.5, 2.0],
    orientationRange = [0.0, Math.PI / 3],
    learningRate = 0.01,
    noiseStd = 0.0,
    scales = null,
    orientations = null,
    phases = null,
  } = {}) {
    this.nModules = nModules;
    this.cellsPerModule = cellsPerModule;
    this.scaleRange = scaleRange;
    this.orientationRange = orientationRange;
    this.learningRate = learningRate;
    this.noiseStd = noiseStd;
    this.nCells = nModules * cellsPerModule;

    // Scales: user-provided or logarithmically spaced (torch.logspace).
    if (scales !== null) {
      this.scales = Float64Array.from(scales);
    } else {
      this.scales = new Float64Array(nModules);
      const a = Math.log10(scaleRange[0]);
      const b = Math.log10(scaleRange[1]);
      for (let i = 0; i < nModules; i++) {
        const t = nModules === 1 ? 0 : i / (nModules - 1);
        this.scales[i] = Math.pow(10, a + (b - a) * t);
      }
    }
    this.scales = this.repeatInterleave(this.scales, cellsPerModule);

    // Orientations: user-provided (degrees if max > 2*pi) or linspace.
    let orients;
    if (orientations !== null) {
      orients = Float64Array.from(orientations);
      let mx = -Infinity;
      for (const v of orients) mx = Math.max(mx, v);
      if (mx > TWO_PI) orients = Float64Array.from(orients, (v) => v * (Math.PI / 180));
    } else {
      orients = new Float64Array(nModules);
      const [lo, hi] = orientationRange;
      for (let i = 0; i < nModules; i++) {
        const t = nModules === 1 ? 0 : i / (nModules - 1);
        orients[i] = lo + (hi - lo) * t;
      }
    }
    this.orientations = this.repeatInterleave(orients, cellsPerModule);

    // Per-cell 2D phase offsets; the caller supplies them (torch.rand in Python).
    if (phases === null) {
      throw new Error('GridCellModule: phases must be provided (length ' + this.nCells * 2 + ')');
    }
    this.gridPhase = Float64Array.from(phases);
    if (this.gridPhase.length !== this.nCells * 2) {
      throw new Error('GridCellModule: phases length ' + this.gridPhase.length + ' != ' + this.nCells * 2);
    }

    this.waveVectors = this.precomputeWaveVectors();
    this.gridPosition = [0, 0];
    this.gridActivation = new Float64Array(this.nCells);
  }

  repeatInterleave(arr, n) {
    const out = new Float64Array(arr.length * n);
    for (let i = 0; i < arr.length; i++) for (let j = 0; j < n; j++) out[i * n + j] = arr[i];
    return out;
  }

  // k_i = (2*pi/lambda) * [cos(theta + i*60deg), sin(theta + i*60deg)], i = 0..2
  precomputeWaveVectors() {
    const wv = new Float64Array(this.nCells * 3 * 2);
    for (let c = 0; c < this.nCells; c++) {
      const kMag = TWO_PI / this.scales[c];
      for (let i = 0; i < 3; i++) {
        const angle = this.orientations[c] + i * (Math.PI / 3);
        wv[(c * 3 + i) * 2] = kMag * Math.cos(angle);
        wv[(c * 3 + i) * 2 + 1] = kMag * Math.sin(angle);
      }
    }
    return wv;
  }

  // activation = ((1/3) * sum_i cos(k_i . (x - phase)) + 1) / 2
  computeActivation(position) {
    const [x, y] = position;
    const out = new Float64Array(this.nCells);
    for (let c = 0; c < this.nCells; c++) {
      const px = x - this.gridPhase[c * 2];
      const py = y - this.gridPhase[c * 2 + 1];
      let sum = 0;
      for (let i = 0; i < 3; i++) {
        const kx = this.waveVectors[(c * 3 + i) * 2];
        const ky = this.waveVectors[(c * 3 + i) * 2 + 1];
        sum += Math.cos(kx * px + ky * py);
      }
      out[c] = (sum / 3 + 1) / 2;
    }
    return out;
  }

  encodePosition(position) {
    return this.computeActivation(position);
  }

  // Dead reckoning: position += velocity * dt
  pathIntegrate(velocity, dt) {
    this.gridPosition[0] += velocity[0] * dt;
    this.gridPosition[1] += velocity[1] * dt;
  }

  // Hebbian-like phase correction toward a sensory activation pattern.
  anchorToInput(sensoryActivation, learningRate = null) {
    const lr = learningRate === null ? this.learningRate : learningRate;
    for (let c = 0; c < this.nCells; c++) {
      const error = sensoryActivation[c] - this.gridActivation[c];
      this.gridPhase[c * 2] = (this.gridPhase[c * 2] + lr * error * this.gridPosition[0]) % TWO_PI;
      this.gridPhase[c * 2 + 1] = (this.gridPhase[c * 2 + 1] + lr * error * this.gridPosition[1]) % TWO_PI;
    }
  }

  forward() {
    this.gridActivation = this.computeActivation(this.gridPosition);
  }
}

// ---------------------------------------------------------------------------
// DisplacementCellModule  (grid_cells.py)
// ---------------------------------------------------------------------------

export class DisplacementCellModule {
  constructor({
    nCells = 100,
    maxDisplacement = 10.0,
    sigma = 0.5,
    nDirections = null,
    nDistances = null,
    maxDistance = null,
  } = {}) {
    this.nCells = nCells;
    this.maxDisplacement = maxDisplacement;
    this.sigma = sigma;

    if (nDirections !== null && nDistances !== null) this.nCells = nDirections * nDistances;
    if (maxDistance !== null) this.maxDisplacement = maxDistance;

    let nMagnitudes;
    let nDirs;
    if (nDirections !== null && nDistances !== null) {
      nMagnitudes = nDistances;
      nDirs = nDirections;
    } else {
      nMagnitudes = Math.floor(Math.sqrt(this.nCells));
      nDirs = Math.floor(this.nCells / nMagnitudes);
    }

    const magnitudes = linspace(0, this.maxDisplacement, nMagnitudes);
    const directions = linspace(0, TWO_PI, nDirs + 1).slice(0, nDirs); // drop 2*pi

    // meshgrid(indexing="ij"): magnitude is the outer (row) index.
    const dx = new Float64Array(this.nCells);
    const dy = new Float64Array(this.nCells);
    let k = 0;
    for (let m = 0; m < nMagnitudes; m++) {
      for (let d = 0; d < nDirs; d++) {
        if (k >= this.nCells) break;
        dx[k] = magnitudes[m] * Math.cos(directions[d]);
        dy[k] = magnitudes[m] * Math.sin(directions[d]);
        k++;
      }
    }
    this.preferredDx = dx;
    this.preferredDy = dy;
    this.displacementActivation = new Float64Array(this.nCells);
    this.previousLocation = null;
  }

  // Gaussian tuning: exp(-dist^2 / (2 sigma^2))
  encodeDisplacement(displacement) {
    const [dx, dy] = displacement;
    const out = new Float64Array(this.nCells);
    const inv = 1 / (2 * this.sigma * this.sigma);
    for (let c = 0; c < this.nCells; c++) {
      const ddx = this.preferredDx[c] - dx;
      const ddy = this.preferredDy[c] - dy;
      out[c] = Math.exp(-(ddx * ddx + ddy * ddy) * inv);
    }
    return out;
  }

  // Population vector decoding: weighted average of preferred displacements.
  decodeDisplacement(activation) {
    let sum = 0;
    for (const a of activation) sum += a;
    if (sum < 1e-8) return [0, 0];
    let ox = 0;
    let oy = 0;
    for (let c = 0; c < this.nCells; c++) {
      const w = activation[c] / sum;
      ox += w * this.preferredDx[c];
      oy += w * this.preferredDy[c];
    }
    return [ox, oy];
  }

  computeDisplacement(fromLocation, toLocation) {
    return this.encodeDisplacement([toLocation[0] - fromLocation[0], toLocation[1] - fromLocation[1]]);
  }

  forward() {
    if (this.gridPosition === undefined) return;
    const current = this.gridPosition;
    if (this.previousLocation !== null) {
      this.displacementActivation = this.encodeDisplacement([
        current[0] - this.previousLocation[0],
        current[1] - this.previousLocation[1],
      ]);
    }
    this.previousLocation = [current[0], current[1]];
  }
}

function linspace(a, b, n) {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = a;
    return out;
  }
  for (let i = 0; i < n; i++) out[i] = a + (b - a) * (i / (n - 1));
  return out;
}

// ---------------------------------------------------------------------------
// LocationEncoder  (grid_cells.py)
// ---------------------------------------------------------------------------

export class LocationEncoder {
  constructor({ locationSize = 256, gridWeight = 0.5, sparsity = 0.02 } = {}) {
    this.locationSize = locationSize;
    this.gridWeight = gridWeight;
    this.sparsity = sparsity;
    this.nActive = Math.max(1, Math.floor(locationSize * sparsity));
    this.locationRepresentation = new Float64Array(locationSize);
    this.activeLocations = new Array(this.nActive).fill(0);
  }

  // Winner-take-all over the grid/sensory blend.
  encodeLocation(gridActivation = null, sensoryLocation = null) {
    const combined = new Float64Array(this.locationSize);
    if (gridActivation !== null) {
      const nGrid = Math.min(gridActivation.length, this.locationSize);
      for (let i = 0; i < nGrid; i++) combined[i] += this.gridWeight * gridActivation[i];
    }
    if (sensoryLocation !== null) {
      for (let i = 0; i < this.locationSize; i++) combined[i] += (1 - this.gridWeight) * sensoryLocation[i];
    }

    let total = 0;
    for (const v of combined) total += v;
    if (total > 0) {
      const top = topKIndices(combined, this.nActive);
      this.activeLocations = top;
      const sparse = new Float64Array(this.locationSize);
      for (const idx of top) sparse[idx] = 1.0;
      return sparse;
    }
    return combined;
  }
}

// Returns indices of the k largest values (ties broken by lower index first).
function topKIndices(arr, k) {
  const idx = arr.map((_, i) => i);
  idx.sort((a, b) => (arr[b] === arr[a] ? a - b : arr[b] - arr[a]));
  return idx.slice(0, k);
}

// ---------------------------------------------------------------------------
// DendriticSegment  (active_dendrites.py)
// ---------------------------------------------------------------------------

export class DendriticSegment {
  constructor({
    nSegments = 128,
    nCells = 256,
    nSynapsesPerSegment = 20,
    activationThreshold = 10,
    learningThreshold = 8,
    permanenceIncrement = 0.1,
    permanenceDecrement = 0.02,
    initialPermanence = 0.21,
    connectedPermanence = 0.2,
    maxNewSynapses = 5,
  } = {}) {
    this.nSegments = nSegments;
    this.nCells = nCells;
    this.nSynapses = nSynapsesPerSegment;
    this.activationThreshold = activationThreshold;
    this.learningThreshold = learningThreshold;
    this.permInc = permanenceIncrement;
    this.permDec = permanenceDecrement;
    this.initialPerm = initialPermanence;
    this.connectedPerm = connectedPermanence;
    this.maxNewSynapses = maxNewSynapses;

    const total = nCells * nSegments;
    this.segmentPermanences = new Float64Array(total * nSynapsesPerSegment);
    this.segmentTargets = new Int32Array(total * nSynapsesPerSegment).fill(-1);
    this.segmentActivation = new Float64Array(total);
    this.segmentActive = new Uint8Array(total);
    this.matchingSegments = new Uint8Array(total);
    this.depolarization = new Float64Array(nCells);
  }

  // A segment's activation is the count of connected synapses whose
  // presynaptic cells are active.
  computeSegmentActivation(presynapticActivity) {
    const { nSynapses } = this;
    for (let s = 0; s < this.segmentTargets.length / nSynapses; s++) {
      let count = 0;
      for (let j = 0; j < nSynapses; j++) {
        const t = this.segmentTargets[s * nSynapses + j];
        if (t >= 0 && this.segmentPermanences[s * nSynapses + j] >= this.connectedPerm && presynapticActivity[t]) {
          count++;
        }
      }
      this.segmentActivation[s] = count;
      this.segmentActive[s] = count >= this.activationThreshold ? 1 : 0;
      this.matchingSegments[s] = count >= this.learningThreshold ? 1 : 0;
    }
  }

  // Depolarization = best segment's activation / threshold, clamped to 1.
  computeDepolarization() {
    for (let c = 0; c < this.nCells; c++) {
      let mx = 0;
      for (let s = 0; s < this.nSegments; s++) {
        mx = Math.max(mx, this.segmentActivation[c * this.nSegments + s]);
      }
      this.depolarization[c] = Math.min(1.0, mx / this.activationThreshold);
    }
  }
}

// ---------------------------------------------------------------------------
// ActiveDendriteComputation  (active_dendrites.py)
// ---------------------------------------------------------------------------

export class ActiveDendriteComputation {
  constructor({
    nmdaThreshold = 0.5,
    nmdaGain = 2.0,
    integrationTau = 10.0,
    calciumTau = 20.0,
    backpropStrength = 0.3,
  } = {}) {
    this.nmdaThreshold = nmdaThreshold;
    this.nmdaGain = nmdaGain;
    this.integrationTau = integrationTau;
    this.calciumTau = calciumTau;
    this.backpropStrength = backpropStrength;
  }

  // f(x) = x + x * gain * sigmoid((x - threshold) * 10)
  nmdaNonlinearity(x) {
    const steepness = 10.0;
    return x + x * this.nmdaGain * sigmoid((x - this.nmdaThreshold) * steepness);
  }

  // One integration step. State lives on this instance (per-neuron scalars).
  computeDendriticIntegration(state, dt, proximalInput, distalInput = null, apicalInput = null) {
    state.proximal += (-state.proximal + proximalInput) * (dt / (this.integrationTau / 2));
    if (distalInput !== null) {
      state.distal += (-state.distal + distalInput) * (dt / this.integrationTau);
    }
    const distalContribution = this.nmdaNonlinearity(state.distal);
    if (apicalInput !== null) {
      state.apical += (-state.apical + apicalInput) * (dt / (this.integrationTau * 2));
    }
    const apicalContribution = this.nmdaNonlinearity(state.apical);

    state.dendriticSpike =
      distalContribution > this.nmdaThreshold * 2 || apicalContribution > this.nmdaThreshold * 2 ? 1 : 0;

    const calciumInflux = state.dendriticSpike * 0.5;
    state.calcium += (-state.calcium + calciumInflux) * (dt / this.calciumTau);

    return state.proximal * (1.0 + distalContribution + apicalContribution);
  }

  applyBackprop(state, somaticSpike) {
    const backpropEffect = somaticSpike * this.backpropStrength;
    state.calcium += backpropEffect * 0.3;
    const combined = state.distal + backpropEffect;
    if (combined > this.nmdaThreshold * 1.5) state.dendriticSpike = 1;
  }
}

export function makeDendriteState() {
  return { proximal: 0, distal: 0, apical: 0, calcium: 0, dendriticSpike: 0 };
}

// ---------------------------------------------------------------------------
// ContextualPrediction  (active_dendrites.py)
// ---------------------------------------------------------------------------

export class ContextualPrediction {
  constructor({
    nCells,
    depolarizationThreshold = 0.5,
    predictionWindow = 10,
    burstThreshold = 0.3,
  } = {}) {
    if (nCells === undefined) throw new Error('ContextualPrediction: nCells is required');
    this.nCells = nCells;
    this.depolThreshold = depolarizationThreshold;
    this.predWindow = predictionWindow;
    this.burstThreshold = burstThreshold;
    this.predictive = new Uint8Array(nCells);
    this.predictionAge = new Int32Array(nCells);
    this.bursting = new Uint8Array(nCells);
  }

  // Neurons become predictive when depolarization exceeds the threshold.
  updatePredictions(contextInput, depolarization = null) {
    const depol = depolarization !== null ? depolarization : contextInput;
    for (let c = 0; c < this.nCells; c++) {
      const newly = depol[c] >= this.depolThreshold ? 1 : 0;
      if (newly) this.predictionAge[c] = 0;
      else if (this.predictive[c]) this.predictionAge[c] += 1;
      const expired = this.predictionAge[c] >= this.predWindow ? 1 : 0;
      this.predictive[c] = (this.predictive[c] | newly) & ~expired;
    }
  }

  // Predicted cells activate normally; unpredicted cells burst (x1.5).
  processActivation(feedforwardInput) {
    const activation = Float64Array.from(feedforwardInput);
    for (let c = 0; c < this.nCells; c++) {
      const would = feedforwardInput[c] >= this.burstThreshold ? 1 : 0;
      const bursting = would & ~this.predictive[c];
      this.bursting[c] = bursting;
      if (bursting) activation[c] *= 1.5;
    }
    return { activation, bursting: this.bursting };
  }
}

// ---------------------------------------------------------------------------
// ColumnVoting / ConsensusNetwork / SDROverlap  (voting.py)
// ---------------------------------------------------------------------------

export class ColumnVoting {
  constructor({
    nObjects = 100,
    representationSize = 2048,
    sparsity = 0.02,
    votingThreshold = 0.5,
    decayRate = 0.1,
    minVotesForConsensus = 3,
  } = {}) {
    this.nObjects = nObjects;
    this.representationSize = representationSize;
    this.sparsity = sparsity;
    this.votingThreshold = votingThreshold;
    this.decayRate = decayRate;
    this.minVotes = minVotesForConsensus;
    this.nActive = Math.max(1, Math.floor(representationSize * sparsity));
    this.voteAccumulator = new Float64Array(nObjects);
    this.objectHypotheses = new Uint8Array(nObjects);
    this.hypothesisConfidence = new Float64Array(nObjects);
    this.consensusReached = false;
    this.consensusObject = -1;
  }

  // votes = overlap / n_local_active, zeroed below the voting threshold.
  castVote(localRepresentation, objectMemories) {
    let nLocalActive = 0;
    for (const v of localRepresentation) if (v > 0) nLocalActive++;
    const votes = new Float64Array(this.nObjects);
    if (nLocalActive === 0) return votes;
    for (let o = 0; o < this.nObjects; o++) {
      let overlap = 0;
      const mem = objectMemories[o];
      for (let i = 0; i < mem.length; i++) if (localRepresentation[i] > 0 && mem[i] > 0) overlap++;
      let vote = overlap / nLocalActive;
      votes[o] = vote >= this.votingThreshold ? vote : 0;
    }
    return votes;
  }

  receiveVotes(incomingVotes, sourceConfidence = 1.0) {
    for (let o = 0; o < this.nObjects; o++) {
      this.voteAccumulator[o] =
        this.voteAccumulator[o] * (1 - this.decayRate) + incomingVotes[o] * sourceConfidence;
      this.objectHypotheses[o] = this.voteAccumulator[o] > 0 ? 1 : 0;
      this.hypothesisConfidence[o] = this.voteAccumulator[o];
    }
  }

  checkConsensus(nColumns) {
    let total = 0;
    for (const v of this.voteAccumulator) total += v;
    if (total === 0) {
      this.consensusReached = false;
      return false;
    }
    const indexed = Array.from(this.voteAccumulator).map((v, i) => [v, i]);
    indexed.sort((a, b) => b[0] - a[0]);
    const topVotes = indexed[0][0];
    const secondVotes = indexed.length > 1 ? indexed[1][0] : 0;
    const marginThreshold = 1.5;
    const consensus =
      topVotes >= this.minVotes && (secondVotes === 0 || topVotes / secondVotes >= marginThreshold);
    this.consensusReached = consensus;
    this.consensusObject = consensus ? indexed[0][1] : -1;
    return consensus;
  }

  resetVoting() {
    this.voteAccumulator.fill(0);
    this.objectHypotheses.fill(0);
    this.hypothesisConfidence.fill(0);
    this.consensusReached = false;
    this.consensusObject = -1;
  }

  forward() {
    for (let o = 0; o < this.nObjects; o++) {
      this.voteAccumulator[o] *= 1 - this.decayRate;
    }
  }
}

export class ConsensusNetwork {
  constructor({ convergenceThreshold = 0.8, maxIterations = 20, broadcastInterval = 1 } = {}) {
    this.convergenceThreshold = convergenceThreshold;
    this.maxIterations = maxIterations;
    this.broadcastInterval = broadcastInterval;
    this.consensusIteration = 0;
    this.networkConsensus = false;
  }

  // Average each column's confidence and broadcast it back to every column.
  broadcastVotes(columns) {
    const n = columns.length;
    if (n < 2) return;
    const avg = new Float64Array(columns[0].nObjects);
    for (const col of columns) {
      for (let o = 0; o < avg.length; o++) avg[o] += col.hypothesisConfidence[o];
    }
    for (let o = 0; o < avg.length; o++) avg[o] /= n;
    for (const col of columns) col.receiveVotes(avg);
  }

  checkNetworkConsensus(columns) {
    if (!columns.length) return false;
    const counts = {};
    for (const col of columns) {
      if (col.consensusReached) counts[col.consensusObject] = (counts[col.consensusObject] || 0) + 1;
    }
    let best = -1;
    let bestCount = 0;
    for (const k in counts) {
      if (counts[k] > bestCount) {
        bestCount = counts[k];
        best = k;
      }
    }
    if (bestCount === 0) return false;
    this.networkConsensus = bestCount / columns.length >= this.convergenceThreshold;
    return this.networkConsensus;
  }

  forward() {
    this.consensusIteration += 1;
  }
}

export const SDROverlap = {
  computeOverlap(a, b) {
    let n = 0;
    for (let i = 0; i < a.length; i++) if (a[i] > 0 && b[i] > 0) n++;
    return n;
  },
  computeOverlapRatio(a, b) {
    let aa = 0;
    let ab = 0;
    for (const v of a) if (v > 0) aa++;
    for (const v of b) if (v > 0) ab++;
    if (aa === 0 || ab === 0) return 0.0;
    let overlap = 0;
    for (let i = 0; i < a.length; i++) if (a[i] > 0 && b[i] > 0) overlap++;
    return overlap / Math.min(aa, ab);
  },
  computeJaccard(a, b) {
    let inter = 0;
    let union = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i] > 0;
      const y = b[i] > 0;
      if (x && y) inter++;
      if (x || y) union++;
    }
    return union === 0 ? 0.0 : inter / union;
  },
};

// ---------------------------------------------------------------------------
// Predictive coding  (predictive_coding.py)
//
// PredictiveCodingLevel mirrors the wiring of Example/numenta/predictive_coding.py:
// TopDownPrediction -> ErrorUnit -> PrecisionWeighting
//   -> FreeEnergyMinimization -> PredictiveCodingLearning
// in that priority order, one forward() per step.
// ---------------------------------------------------------------------------

export const defaultPredictiveConfig = {
  predictionTau: 20.0,
  errorTau: 10.0,
  precisionTau: 100.0,
  learningRate: 0.01,
  precisionLearningRate: 0.001,
  minPrecision: 0.01,
  maxPrecision: 100.0,
  errorNonlinearity: 'relu',
  useSparseErrors: true,
  errorThreshold: 0.1,
  temporalSmoothing: 0.9,
};

export class PredictiveCodingLevel {
  constructor(size, config = {}) {
    this.size = size;
    this.cfg = Object.assign({}, defaultPredictiveConfig, config);
    this.activity = new Float64Array(size);
    this.topDownInput = new Float64Array(size);
    this.topDownPrediction = new Float64Array(size);
    // top_down_weights = eye * 0.5
    this.topDownWeights = new Float64Array(size * size);
    for (let i = 0; i < size; i++) this.topDownWeights[i * size + i] = 0.5;
    this.prediction = new Float64Array(size);
    this.signedError = new Float64Array(size);
    this.predictionError = new Float64Array(size);
    this.errorMagnitude = new Float64Array(size);
    this.precision = new Float64Array(size).fill(1);
    this.precisionEstimate = new Float64Array(size).fill(1);
    this.errorVariance = new Float64Array(size).fill(1);
    this.sensoryPrecision = new Float64Array(size).fill(1);
    this.priorPrecision = new Float64Array(size).fill(1);
    this.attentionGain = new Float64Array(size).fill(1);
    this.effectivePrecision = new Float64Array(size).fill(1);
    this.precisionGrad = new Float64Array(size);
    this.freeEnergy = 0;
    this.accuracyTerm = 0;
    this.complexityTerm = 0;
  }

  // TopDownPrediction.forward: prediction = W^T @ top_down_input
  topDown() {
    const { size, topDownWeights: w, topDownInput: td, topDownPrediction: pred } = this;
    for (let i = 0; i < size; i++) {
      let s = 0;
      for (let j = 0; j < size; j++) s += w[j * size + i] * td[j]; // W^T row i
      pred[i] = s;
    }
    this.prediction = pred;
  }

  // ErrorUnit.forward
  error() {
    const { size, activity, prediction, cfg } = this;
    for (let i = 0; i < size; i++) {
      const e = activity[i] - prediction[i];
      this.signedError[i] = e;
      let pe =
        cfg.errorNonlinearity === 'relu' ? Math.max(0, e)
        : cfg.errorNonlinearity === 'tanh' ? Math.tanh(e)
        : e;
      this.errorMagnitude[i] = Math.abs(e);
      this.predictionError[i] = pe;
    }
    if (cfg.useSparseErrors) {
      for (let i = 0; i < size; i++) {
        if (this.errorMagnitude[i] <= cfg.errorThreshold) this.predictionError[i] = 0;
      }
    }
    this.updatePrecision();
  }

  // ErrorUnit._update_precision: EMA of squared error, precision = 1/variance
  updatePrecision() {
    const { size, cfg } = this;
    const alpha = 1.0 / cfg.precisionTau; // dt = 1 s in the golden run
    for (let i = 0; i < size; i++) {
      const eSq = this.signedError[i] * this.signedError[i];
      this.errorVariance[i] = (1 - alpha) * this.errorVariance[i] + alpha * eSq;
      this.precisionEstimate[i] = 1.0 / (this.errorVariance[i] + 1e-8);
      this.precision[i] = clamp(this.precisionEstimate[i], cfg.minPrecision, cfg.maxPrecision);
    }
  }

  // PrecisionWeighting.forward (attention_modulated=False in the golden run)
  precisionWeighting() {
    const { size, cfg } = this;
    for (let i = 0; i < size; i++) {
      this.effectivePrecision[i] = this.sensoryPrecision[i] * this.priorPrecision[i];
      this.effectivePrecision[i] = clamp(this.effectivePrecision[i], cfg.minPrecision, cfg.maxPrecision);
      this.precision[i] = this.effectivePrecision[i];
    }
    this.learnPrecision();
  }

  // PrecisionWeighting._learn_precision: gradient descent on F
  learnPrecision() {
    const { size, cfg } = this;
    const lr = cfg.precisionLearningRate * 1.0; // dt = 1 s
    for (let i = 0; i < size; i++) {
      const eSq = this.signedError[i] * this.signedError[i];
      this.precisionGrad[i] = 0.5 * (eSq - 1.0 / (this.sensoryPrecision[i] + 1e-8));
      this.sensoryPrecision[i] = clamp(
        this.sensoryPrecision[i] - lr * this.precisionGrad[i],
        cfg.minPrecision,
        cfg.maxPrecision,
      );
    }
  }

  // FreeEnergyMinimization.forward
  computeFreeEnergy() {
    const { size } = this;
    let sumPE = 0;
    let sumLog = 0;
    for (let i = 0; i < size; i++) {
      sumPE += this.precision[i] * this.signedError[i] * this.signedError[i];
      sumLog += Math.log(this.precision[i] + 1e-8);
    }
    this.accuracyTerm = 0.5 * sumPE;
    this.complexityTerm = -0.5 * sumLog;
    this.freeEnergy = this.accuracyTerm + this.complexityTerm;
  }

  // PredictiveCodingLearning.forward (use_hebbian=False in the golden run)
  learning(useHebbian = false, weightDecay = 0.0001) {
    const { size, cfg } = this;
    const lr = cfg.learningRate * 1.0; // dt = 1 s
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        let dw = this.topDownInput[i] * this.predictionError[j];
        if (useHebbian) dw += 0.1 * this.activity[i] * this.activity[j];
        this.topDownWeights[i * size + j] += lr * (dw - weightDecay * this.topDownWeights[i * size + j]);
      }
    }
  }

  // One full step in the example's priority order.
  forward() {
    this.topDown();
    this.error();
    this.precisionWeighting();
    this.computeFreeEnergy();
    this.learning(false);
  }
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// PredictiveCorticalColumn structure  (predictive_hierarchy.py)
//
// Static description of the canonical microcircuit: layer sizes and the
// connections the code actually builds. No simulation.
// ---------------------------------------------------------------------------

export function corticalColumnStructure(columnSize = 100) {
  const layerSizes = {
    L4: columnSize,
    L23: Math.floor(columnSize * 1.5),
    L5: Math.floor(columnSize * 0.8),
    L6: Math.floor(columnSize * 0.5),
  };
  const connections = [
    { from: 'L4', to: 'L23', type: 'feedforward', label: 'error propagation' },
    { from: 'L23', to: 'L4', type: 'feedback', label: 'predictions' },
    { from: 'L23', to: 'L5', type: 'feedforward', label: 'output generation' },
    { from: 'L6', to: 'L4', type: 'lateral', label: 'precision modulation' },
  ];
  return { columnSize, layerSizes, connections };
}

// ---------------------------------------------------------------------------
// HierarchicalPredictiveCoding  (predictive_coding.py)
//
// One population carries the whole hierarchy: level_predictions/errors/
// precisions plus feedforward/feedback weight matrices between levels.
// Weights are passed in (torch.randn * 0.1 in Python) so runs are reproducible.
// ---------------------------------------------------------------------------

export class HierarchicalPredictiveCoding {
  constructor(levelSizes, feedforwardWeights, feedbackWeights) {
    this.levelSizes = levelSizes;
    this.feedforwardWeights = feedforwardWeights; // [i]: (size_i x size_{i+1}) row-major
    this.feedbackWeights = feedbackWeights; // [i]: (size_{i+1} x size_i) row-major
    this.levelPredictions = levelSizes.map((s) => new Float64Array(s));
    this.levelErrors = levelSizes.map((s) => new Float64Array(s));
    this.levelPrecisions = levelSizes.map((s) => new Float64Array(s).fill(1));
  }

  // Bottom-up: errors up (precision-weighted); top-down: predictions down.
  forward(sensory) {
    const n = this.levelSizes.length;
    let current = Float64Array.from(sensory.slice(0, this.levelSizes[0]));
    for (let i = 0; i < n - 1; i++) {
      const size = this.levelSizes[i];
      for (let j = 0; j < size; j++) {
        this.levelErrors[i][j] = current[j] - this.levelPredictions[i][j];
      }
      const w = this.feedforwardWeights[i];
      const next = new Float64Array(this.levelSizes[i + 1]);
      for (let j = 0; j < this.levelSizes[i + 1]; j++) {
        let s = 0;
        for (let k = 0; k < size; k++) {
          s += w[k * this.levelSizes[i + 1] + j] * this.levelErrors[i][k] * this.levelPrecisions[i][k]; // W^T
        }
        next[j] = s;
      }
      current = next;
    }
    for (let i = n - 2; i >= 0; i--) {
      const above = i < n - 1 ? this.levelPredictions[i + 1] : current;
      const w = this.feedbackWeights[i];
      for (let j = 0; j < this.levelSizes[i]; j++) {
        let s = 0;
        for (let k = 0; k < above.length; k++) {
          s += w[k * this.levelSizes[i] + j] * above[k]; // W^T
        }
        this.levelPredictions[i][j] = s;
      }
    }
  }
}
