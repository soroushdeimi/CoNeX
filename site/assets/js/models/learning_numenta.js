// Pure math models for Numenta learning rules in CoNeX:
// 1. Dendritic segment learning (DendriticSegment.learn_on_segments,
//    TemporalMemory SegmentStorage adapt_segment, punish_segments, grow_synapses)
// 2. PredictiveSynapseLearning (feedforward/feedback predictive synapses with eligibility traces)
// 3. Precision rule _learn_precision (gradient descent on Gaussian free energy)
// 4. PredictiveCodingNetwork (online predictive coding with weights heatmap, precision, error)

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// 1. Dendritic segment learning
// ---------------------------------------------------------------------------

export class DendriticSegmentModel {
  constructor(config = {}) {
    const {
      nCells = 1,
      nSegments = 2,
      nSynapsesPerSegment = 4,
      activationThreshold = 2,
      learningThreshold = 1,
      permanenceIncrement = 0.1,
      permanenceDecrement = 0.02,
      initialPermanence = 0.21,
      connectedPermanence = 0.2,
      maxNewSynapses = 2,
    } = config;

    if (!Number.isInteger(nCells) || nCells < 1) {
      throw new Error(`nCells must be a positive integer, got ${nCells}`);
    }
    if (!Number.isInteger(nSegments) || nSegments < 1) {
      throw new Error(`nSegments must be a positive integer, got ${nSegments}`);
    }
    if (!Number.isInteger(nSynapsesPerSegment) || nSynapsesPerSegment < 1) {
      throw new Error(`nSynapsesPerSegment must be a positive integer, got ${nSynapsesPerSegment}`);
    }
    if (!Number.isInteger(activationThreshold) || activationThreshold < 0) {
      throw new Error(`activationThreshold must be a non-negative integer, got ${activationThreshold}`);
    }
    if (!Number.isInteger(learningThreshold) || learningThreshold < 0) {
      throw new Error(`learningThreshold must be a non-negative integer, got ${learningThreshold}`);
    }
    if (typeof permanenceIncrement !== 'number' || permanenceIncrement <= 0 || permanenceIncrement > 1) {
      throw new Error(`permanenceIncrement must be in (0, 1], got ${permanenceIncrement}`);
    }
    if (typeof permanenceDecrement !== 'number' || permanenceDecrement <= 0 || permanenceDecrement > 1) {
      throw new Error(`permanenceDecrement must be in (0, 1], got ${permanenceDecrement}`);
    }
    if (typeof initialPermanence !== 'number' || initialPermanence < 0 || initialPermanence > 1) {
      throw new Error(`initialPermanence must be in [0, 1], got ${initialPermanence}`);
    }
    if (typeof connectedPermanence !== 'number' || connectedPermanence < 0 || connectedPermanence > 1) {
      throw new Error(`connectedPermanence must be in [0, 1], got ${connectedPermanence}`);
    }
    if (!Number.isInteger(maxNewSynapses) || maxNewSynapses < 0) {
      throw new Error(`maxNewSynapses must be a non-negative integer, got ${maxNewSynapses}`);
    }

    this.nCells = nCells;
    this.nSegments = nSegments;
    this.nSynapses = nSynapsesPerSegment;
    this.activationThreshold = activationThreshold;
    this.learningThreshold = learningThreshold;
    this.permInc = permanenceIncrement;
    this.permDec = permanenceDecrement;
    this.initialPerm = initialPermanence;
    this.connectedPerm = connectedPermanence;
    this.maxNewSynapses = maxNewSynapses;

    const totalSegs = nCells * nSegments;
    // Synapse targets: Int32Array(totalSegs * nSynapses), -1 = unconnected
    this.targets = new Int32Array(totalSegs * this.nSynapses).fill(-1);
    // Synapse permanences: Float64Array(totalSegs * nSynapses)
    this.permanences = new Float64Array(totalSegs * this.nSynapses);

    this.segmentActivation = new Float64Array(totalSegs);
    this.segmentActive = new Array(totalSegs).fill(false);
    this.matchingSegments = new Array(totalSegs).fill(false);
    this.depolarization = new Float64Array(nCells);
  }

  setSegment(segIdx, targets, perms) {
    const offset = segIdx * this.nSynapses;
    for (let i = 0; i < this.nSynapses; i++) {
      this.targets[offset + i] = targets[i] !== undefined ? targets[i] : -1;
      this.permanences[offset + i] = perms[i] !== undefined ? perms[i] : 0.0;
    }
  }

  getSegment(segIdx) {
    const offset = segIdx * this.nSynapses;
    const tgts = [];
    const prms = [];
    for (let i = 0; i < this.nSynapses; i++) {
      tgts.push(this.targets[offset + i]);
      prms.push(this.permanences[offset + i]);
    }
    return { targets: tgts, permanences: prms };
  }

  computeActivation(presynapticActivity) {
    const totalSegs = this.nCells * this.nSegments;
    for (let s = 0; s < totalSegs; s++) {
      const offset = s * this.nSynapses;
      let count = 0;
      for (let syn = 0; syn < this.nSynapses; syn++) {
        const tgt = this.targets[offset + syn];
        const perm = this.permanences[offset + syn];
        const connected = perm >= this.connectedPerm;
        const valid = tgt >= 0 && tgt < presynapticActivity.length;
        const active = valid && Boolean(presynapticActivity[tgt]);
        if (connected && active) {
          count++;
        }
      }
      this.segmentActivation[s] = count;
      this.segmentActive[s] = count >= this.activationThreshold;
      this.matchingSegments[s] = count >= this.learningThreshold;
    }

    this.computeDepolarization();
    return {
      activation: Array.from(this.segmentActivation),
      active: [...this.segmentActive],
      matching: [...this.matchingSegments],
      depolarization: Array.from(this.depolarization),
    };
  }

  computeDepolarization() {
    for (let c = 0; c < this.nCells; c++) {
      let maxAct = 0;
      for (let s = 0; s < this.nSegments; s++) {
        const segIdx = c * this.nSegments + s;
        if (this.segmentActivation[segIdx] > maxAct) {
          maxAct = this.segmentActivation[segIdx];
        }
      }
      this.depolarization[c] = Math.min(1.0, maxAct / (this.activationThreshold || 1));
    }
  }

  learnOnSegments({ activeCells, winnerCells, presynapticActivity, rng, newTargets }) {
    const totalSegs = this.nCells * this.nSegments;
    const actions = [];

    // Find active presynaptic cell indices
    const activePresynaptic = [];
    for (let i = 0; i < presynapticActivity.length; i++) {
      if (presynapticActivity[i]) activePresynaptic.push(i);
    }

    for (let s = 0; s < totalSegs; s++) {
      const cellIdx = Math.floor(s / this.nSegments);
      const isWinner = Boolean(winnerCells[cellIdx]);
      const matched = Boolean(this.matchingSegments[s]);
      const shouldLearn = isWinner && matched;

      const segActions = [];
      const offset = s * this.nSynapses;

      if (!shouldLearn) {
        for (let syn = 0; syn < this.nSynapses; syn++) {
          segActions.push({
            segmentIdx: s,
            synapseIdx: syn,
            target: this.targets[offset + syn],
            permBefore: this.permanences[offset + syn],
            permAfter: this.permanences[offset + syn],
            delta: 0,
            action: 'inactive_segment',
            reason: !isWinner ? 'cell is not winner' : 'segment below learning threshold',
          });
        }
        actions.push(segActions);
        continue;
      }

      // Reinforce active synapses, weaken inactive ones
      const emptySlots = [];
      for (let syn = 0; syn < this.nSynapses; syn++) {
        const tgt = this.targets[offset + syn];
        const permBefore = this.permanences[offset + syn];

        if (tgt < 0) {
          emptySlots.push(syn);
          segActions.push({
            segmentIdx: s,
            synapseIdx: syn,
            target: tgt,
            permBefore,
            permAfter: permBefore,
            delta: 0,
            action: 'empty',
            reason: 'unconnected slot',
          });
          continue;
        }

        const valid = tgt < presynapticActivity.length;
        const isActive = valid && Boolean(presynapticActivity[tgt]);

        let permAfter = permBefore;
        let delta = 0;
        let action = 'unchanged';
        let reason = '';

        if (isActive) {
          permAfter = clamp(permBefore + this.permInc, 0.0, 1.0);
          delta = permAfter - permBefore;
          action = 'reinforced';
          reason = `presynaptic neuron ${tgt} was active (reinforced +${this.permInc})`;
        } else {
          permAfter = clamp(permBefore - this.permDec, 0.0, 1.0);
          delta = permAfter - permBefore;
          action = 'punished';
          reason = `presynaptic neuron ${tgt} was inactive (punished -${this.permDec})`;
        }

        this.permanences[offset + syn] = permAfter;
        segActions.push({
          segmentIdx: s,
          synapseIdx: syn,
          target: tgt,
          permBefore,
          permAfter,
          delta,
          action,
          reason,
        });
      }

      // Grow new synapses into empty slots if room
      if (emptySlots.length > 0 && activePresynaptic.length > 0 && this.maxNewSynapses > 0) {
        const nToGrow = Math.min(emptySlots.length, this.maxNewSynapses, activePresynaptic.length);
        // Select presynaptic candidates (using newTargets if provided, else rng/activePresynaptic)
        let candidates = newTargets || [...activePresynaptic];
        if (!newTargets && rng && typeof rng.next === 'function') {
          // Fisher-Yates shuffle
          for (let i = candidates.length - 1; i > 0; i--) {
            const j = Math.floor(rng.next() * (i + 1));
            [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
          }
        }
        for (let g = 0; g < nToGrow; g++) {
          const slot = emptySlots[g];
          const newTarget = candidates[g % candidates.length];
          this.targets[offset + slot] = newTarget;
          this.permanences[offset + slot] = this.initialPerm;
          segActions[slot] = {
            segmentIdx: s,
            synapseIdx: slot,
            target: newTarget,
            permBefore: 0.0,
            permAfter: this.initialPerm,
            delta: this.initialPerm,
            action: 'grown',
            reason: `new synapse grown to active presynaptic neuron ${newTarget} (init perm ${this.initialPerm})`,
          };
        }
      }

      actions.push(segActions);
    }

    return actions;
  }
}

// ---------------------------------------------------------------------------
// 2. TemporalMemory SegmentStorage operations
// ---------------------------------------------------------------------------

export class SegmentStorageModel {
  constructor(nCells, maxSegments = 255, maxSynapses = 255) {
    if (!Number.isInteger(nCells) || nCells < 1) {
      throw new Error(`nCells must be positive integer, got ${nCells}`);
    }
    this.nCells = nCells;
    this.maxSegments = maxSegments;
    this.maxSynapses = maxSynapses;
    this.segments = new Map(); // key: `${cellIdx},${segIdx}` -> { targets: Int32Array, permanences: Float64Array }
    this.segmentCounts = new Int32Array(nCells);
  }

  setSegment(cellIdx, segIdx, targets, perms) {
    this.segments.set(`${cellIdx},${segIdx}`, {
      targets: Int32Array.from(targets),
      permanences: Float64Array.from(perms),
    });
    this.segmentCounts[cellIdx] = Math.max(this.segmentCounts[cellIdx], segIdx + 1);
  }

  getSegment(cellIdx, segIdx) {
    return this.segments.get(`${cellIdx},${segIdx}`) || null;
  }

  adaptSegment(cellIdx, segmentIdx, activeCells, permInc = 0.1, permDec = 0.02) {
    const seg = this.getSegment(cellIdx, segmentIdx);
    if (!seg) return null;
    const actions = [];
    for (let i = 0; i < seg.targets.length; i++) {
      const tgt = seg.targets[i];
      const permBefore = seg.permanences[i];
      const isActive = tgt >= 0 && tgt < activeCells.length && Boolean(activeCells[tgt]);
      let permAfter = permBefore;
      let action = 'unchanged';
      let reason = '';
      if (isActive) {
        permAfter = clamp(permBefore + permInc, 0.0, 1.0);
        action = 'reinforced';
        reason = `target cell ${tgt} active: reinforced (+${permInc})`;
      } else {
        permAfter = clamp(permBefore - permDec, 0.0, 1.0);
        action = 'punished';
        reason = `target cell ${tgt} inactive: punished (-${permDec})`;
      }
      seg.permanences[i] = permAfter;
      actions.push({
        synapseIdx: i,
        target: tgt,
        permBefore,
        permAfter,
        delta: permAfter - permBefore,
        action,
        reason,
      });
    }
    return actions;
  }

  punishSegments(cellIdx, activeCells, decrement = 0.05, threshold = 1, connectedPerm = 0.2) {
    const punished = [];
    const count = this.segmentCounts[cellIdx];
    for (let segIdx = 0; segIdx < count; segIdx++) {
      const seg = this.getSegment(cellIdx, segIdx);
      if (!seg) continue;
      let nActive = 0;
      for (let i = 0; i < seg.targets.length; i++) {
        const tgt = seg.targets[i];
        const connected = seg.permanences[i] >= connectedPerm;
        const active = tgt >= 0 && tgt < activeCells.length && Boolean(activeCells[tgt]);
        if (connected && active) nActive++;
      }
      if (nActive >= threshold) {
        // Punish active synapses that caused this false prediction
        for (let i = 0; i < seg.targets.length; i++) {
          const tgt = seg.targets[i];
          const active = tgt >= 0 && tgt < activeCells.length && Boolean(activeCells[tgt]);
          if (active) {
            const before = seg.permanences[i];
            const after = clamp(before - decrement, 0.0, 1.0);
            seg.permanences[i] = after;
            punished.push({
              segmentIdx: segIdx,
              synapseIdx: i,
              target: tgt,
              permBefore: before,
              permAfter: after,
              delta: after - before,
              reason: `false prediction punishment: active synapse on unfulfilled segment decremented (-${decrement})`,
            });
          }
        }
      }
    }
    return punished;
  }

  growSynapses(cellIdx, segmentIdx, winnerCells, initialPerm = 0.21, maxNew = 2, maxSynapses = 10, rng = null) {
    const seg = this.getSegment(cellIdx, segmentIdx);
    if (!seg) return [];
    const currentTargets = new Set(Array.from(seg.targets));
    if (currentTargets.size >= maxSynapses) return [];

    const candidates = [];
    for (let i = 0; i < winnerCells.length; i++) {
      if (winnerCells[i] && !currentTargets.has(i)) {
        candidates.push(i);
      }
    }
    if (candidates.length === 0) return [];

    if (rng && typeof rng.next === 'function') {
      for (let i = candidates.length - 1; i > 0; i--) {
        const j = Math.floor(rng.next() * (i + 1));
        [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
      }
    }

    const nNew = Math.min(candidates.length, maxNew, maxSynapses - currentTargets.size);
    const newTargets = candidates.slice(0, nNew);

    const mergedTargets = new Int32Array(seg.targets.length + nNew);
    const mergedPerms = new Float64Array(seg.permanences.length + nNew);

    mergedTargets.set(seg.targets, 0);
    mergedPerms.set(seg.permanences, 0);

    const grown = [];
    for (let g = 0; g < nNew; g++) {
      const idx = seg.targets.length + g;
      mergedTargets[idx] = newTargets[g];
      mergedPerms[idx] = initialPerm;
      grown.push({
        synapseIdx: idx,
        target: newTargets[g],
        permBefore: 0.0,
        permAfter: initialPerm,
        delta: initialPerm,
        action: 'grown',
        reason: `new synapse grown to winner cell ${newTargets[g]} with initial permanence ${initialPerm}`,
      });
    }

    seg.targets = mergedTargets;
    seg.permanences = mergedPerms;
    return grown;
  }
}

// ---------------------------------------------------------------------------
// 3. PredictiveSynapseLearning
// ---------------------------------------------------------------------------

export class PredictiveSynapseModel {
  constructor(config = {}) {
    const {
      srcSize = 3,
      dstSize = 2,
      learningRate = 0.05,
      dt = 1.0,
      learningType = 'error_driven',
      initialWeights = null,
    } = config;

    if (!Number.isInteger(srcSize) || srcSize < 1) {
      throw new Error(`srcSize must be a positive integer, got ${srcSize}`);
    }
    if (!Number.isInteger(dstSize) || dstSize < 1) {
      throw new Error(`dstSize must be a positive integer, got ${dstSize}`);
    }
    if (typeof learningRate !== 'number' || learningRate <= 0) {
      throw new Error(`learningRate must be positive, got ${learningRate}`);
    }
    if (typeof dt !== 'number' || dt <= 0) {
      throw new Error(`dt must be positive, got ${dt}`);
    }
    const validTypes = ['error_driven', 'hebbian', 'combined'];
    if (!validTypes.includes(learningType)) {
      throw new Error(`learningType must be one of ${validTypes.join(', ')}, got ${learningType}`);
    }

    this.srcSize = srcSize;
    this.dstSize = dstSize;
    this.lr = learningRate;
    this.dt = dt;
    this.learningType = learningType;

    const count = srcSize * dstSize;
    this.weights = new Float64Array(count);
    this.eligibility = new Float64Array(count);

    if (initialWeights) {
      for (let i = 0; i < count; i++) {
        this.weights[i] = initialWeights[i] !== undefined ? initialWeights[i] : 0.0;
      }
    }
  }

  step({ srcActivity, dstError, dstActivity }) {
    const { srcSize, dstSize, lr, dt, learningType } = this;
    const effLr = lr * dt;

    for (let i = 0; i < srcSize; i++) {
      const srcA = srcActivity[i] || 0.0;
      for (let j = 0; j < dstSize; j++) {
        const idx = i * dstSize + j;
        const err = dstError ? dstError[j] || 0.0 : 0.0;
        const dstA = dstActivity ? dstActivity[j] || 0.0 : 0.0;

        let deltaW = 0.0;
        if (learningType === 'error_driven') {
          deltaW = srcA * err;
        } else if (learningType === 'hebbian') {
          deltaW = srcA * dstA;
        } else {
          // combined: error_driven + 0.1 * hebbian
          deltaW = srcA * err + 0.1 * srcA * dstA;
        }

        // Update eligibility trace: 0.9 * eligibility + 0.1 * deltaW
        this.eligibility[idx] = 0.9 * this.eligibility[idx] + 0.1 * deltaW;

        // Apply update with eligibility
        this.weights[idx] = this.weights[idx] + effLr * this.eligibility[idx];
      }
    }

    return {
      weights: Array.from(this.weights),
      eligibility: Array.from(this.eligibility),
    };
  }
}

// ---------------------------------------------------------------------------
// 4. Precision rule _learn_precision
// ---------------------------------------------------------------------------

export function learnPrecision({
  sensoryPrecision,
  signedError,
  precisionLearningRate = 0.02,
  minPrecision = 0.01,
  maxPrecision = 100.0,
  dt = 1.0,
}) {
  if (typeof precisionLearningRate !== 'number' || precisionLearningRate <= 0) {
    throw new Error(`precisionLearningRate must be positive, got ${precisionLearningRate}`);
  }
  if (typeof minPrecision !== 'number' || minPrecision <= 0) {
    throw new Error(`minPrecision must be positive, got ${minPrecision}`);
  }
  if (typeof maxPrecision !== 'number' || maxPrecision < minPrecision) {
    throw new Error(`maxPrecision must be >= minPrecision, got ${maxPrecision}`);
  }

  const n = sensoryPrecision.length;
  const lr = precisionLearningRate * dt;
  const grad = new Float64Array(n);
  const updated = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const err = signedError[i] || 0.0;
    const prec = sensoryPrecision[i];
    const eSq = err * err;
    // dF/d_precision = 0.5 * (error^2 - 1 / (precision + 1e-8))
    const g = 0.5 * (eSq - 1.0 / (prec + 1e-8));
    grad[i] = g;
    updated[i] = clamp(prec - lr * g, minPrecision, maxPrecision);
  }

  return {
    precisionGrad: Array.from(grad),
    sensoryPrecision: Array.from(updated),
  };
}

// ---------------------------------------------------------------------------
// 5. PredictiveCodingNetwork (online learning with heatmap, precision, error)
// ---------------------------------------------------------------------------

export class PredictiveCodingNetwork {
  constructor(size, config = {}) {
    if (!Number.isInteger(size) || size < 1) {
      throw new Error(`size must be positive integer, got ${size}`);
    }

    const {
      learningRate = 0.05,
      precisionTau = 20.0,
      precisionLearningRate = 0.01,
      minPrecision = 0.01,
      maxPrecision = 100.0,
      errorNonlinearity = 'linear',
      useSparseErrors = false,
      errorThreshold = 0.1,
      weightDecay = 0.0001,
      useHebbian = false,
      dt = 1.0,
    } = config;

    this.size = size;
    this.lr = learningRate;
    this.precisionTau = precisionTau;
    this.precisionLr = precisionLearningRate;
    this.minPrec = minPrecision;
    this.maxPrec = maxPrecision;
    this.errorNonlin = errorNonlinearity;
    this.useSparseErrors = useSparseErrors;
    this.errorThreshold = errorThreshold;
    this.weightDecay = weightDecay;
    this.useHebbian = useHebbian;
    this.dt = dt;

    this.topDownWeights = new Float64Array(size * size);
    for (let i = 0; i < size; i++) this.topDownWeights[i * size + i] = 0.5;

    this.activity = new Float64Array(size);
    this.topDownInput = new Float64Array(size);

    this.prediction = new Float64Array(size);
    this.signedError = new Float64Array(size);
    this.predictionError = new Float64Array(size);
    this.errorMagnitude = new Float64Array(size);

    this.errorVariance = new Float64Array(size).fill(1.0);
    this.precisionEstimate = new Float64Array(size).fill(1.0);
    this.sensoryPrecision = new Float64Array(size).fill(1.0);
    this.priorPrecision = new Float64Array(size).fill(1.0);
    this.effectivePrecision = new Float64Array(size).fill(1.0);
    this.precision = new Float64Array(size).fill(1.0);
    this.precisionGrad = new Float64Array(size);

    this.freeEnergy = 0.0;
    this.accuracyTerm = 0.0;
    this.complexityTerm = 0.0;
  }

  topDown() {
    const { size, topDownWeights: w, topDownInput: td, prediction: pred } = this;
    for (let i = 0; i < size; i++) {
      let sum = 0.0;
      for (let j = 0; j < size; j++) {
        sum += w[j * size + i] * td[j]; // W^T @ top_down
      }
      pred[i] = sum;
    }
  }

  error() {
    const { size, activity, prediction, errorNonlin, useSparseErrors, errorThreshold } = this;
    for (let i = 0; i < size; i++) {
      const e = activity[i] - prediction[i];
      this.signedError[i] = e;
      let pe = e;
      if (errorNonlin === 'relu') {
        pe = Math.max(0.0, e);
      } else if (errorNonlin === 'tanh') {
        pe = Math.tanh(e);
      }
      this.errorMagnitude[i] = Math.abs(e);
      if (useSparseErrors && this.errorMagnitude[i] <= errorThreshold) {
        pe = 0.0;
      }
      this.predictionError[i] = pe;
    }

    // EMA of error variance
    const alpha = this.dt / this.precisionTau;
    for (let i = 0; i < size; i++) {
      const eSq = this.signedError[i] * this.signedError[i];
      this.errorVariance[i] = (1.0 - alpha) * this.errorVariance[i] + alpha * eSq;
      this.precisionEstimate[i] = 1.0 / (this.errorVariance[i] + 1e-8);
      this.precision[i] = clamp(this.precisionEstimate[i], this.minPrec, this.maxPrec);
    }
  }

  precisionWeighting() {
    const { size, minPrec, maxPrec, precisionLr, dt } = this;
    for (let i = 0; i < size; i++) {
      this.effectivePrecision[i] = clamp(
        this.sensoryPrecision[i] * this.priorPrecision[i],
        minPrec,
        maxPrec,
      );
      this.precision[i] = this.effectivePrecision[i];
    }

    // _learn_precision
    const lr = precisionLr * dt;
    for (let i = 0; i < size; i++) {
      const eSq = this.signedError[i] * this.signedError[i];
      this.precisionGrad[i] = 0.5 * (eSq - 1.0 / (this.sensoryPrecision[i] + 1e-8));
      this.sensoryPrecision[i] = clamp(
        this.sensoryPrecision[i] - lr * this.precisionGrad[i],
        minPrec,
        maxPrec,
      );
    }
  }

  computeFreeEnergy() {
    const { size } = this;
    let sumPE = 0.0;
    let sumLog = 0.0;
    for (let i = 0; i < size; i++) {
      sumPE += this.precision[i] * this.signedError[i] * this.signedError[i];
      sumLog += Math.log(this.precision[i] + 1e-8);
    }
    this.accuracyTerm = 0.5 * sumPE;
    this.complexityTerm = -0.5 * sumLog;
    this.freeEnergy = this.accuracyTerm + this.complexityTerm;
  }

  learning() {
    const { size, lr, dt, topDownInput: td, predictionError: err, useHebbian, weightDecay, activity: act } = this;
    const effLr = lr * dt;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = i * size + j;
        let deltaW = td[i] * err[j];
        if (useHebbian) {
          deltaW += 0.1 * act[i] * act[j];
        }
        if (weightDecay > 0) {
          deltaW -= weightDecay * this.topDownWeights[idx];
        }
        this.topDownWeights[idx] += effLr * deltaW;
      }
    }
  }

  step() {
    this.topDown();
    this.error();
    this.precisionWeighting();
    this.computeFreeEnergy();
    this.learning();

    let sumAbs = 0.0;
    let sumPrec = 0.0;
    for (let i = 0; i < this.size; i++) {
      sumAbs += this.errorMagnitude[i];
      sumPrec += this.precision[i];
    }

    return {
      meanAbsError: sumAbs / this.size,
      meanPrecision: sumPrec / this.size,
      freeEnergy: this.freeEnergy,
      prediction: Array.from(this.prediction),
      signedError: Array.from(this.signedError),
      precision: Array.from(this.precision),
      weights: Array.from(this.topDownWeights),
    };
  }
}
