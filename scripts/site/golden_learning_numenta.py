"""Golden reference numbers for the learning_numenta models of the site.

Runs the real CoNeX classes:
- DendriticSegment and learn_on_segments (active_dendrites.py)
- TemporalMemory SegmentStorage (adapt_segment, punish_segments, grow_synapses)
- PredictiveSynapseLearning (predictive.py)
- PredictiveCodingLearning and PrecisionWeighting._learn_precision (predictive_coding.py)

Writes site/tests/golden/learning_numenta.json.

Run:
    /Users/soroushdeimi/Desktop/myprojects/.conex-site-venv/bin/python scripts/site/golden_learning_numenta.py
"""

import json
import os
import torch
from pymonntorch import Network, NeuronGroup, SynapseGroup

from conex import (
    TimeResolution,
    PredictiveCodingConfig,
    PredictiveSynapseLearning,
    PredictiveCodingLearning,
    PrecisionWeighting,
    ErrorUnit,
    TopDownPrediction,
    FreeEnergyMinimization,
    DendriticSegment,
    DendriticSegmentConfig,
    TemporalMemory,
    TemporalMemoryConfig,
)
from conex.behaviors.synapses.numenta.predictive import PredictiveSynapseInit
from conex.behaviors.neurons.numenta.sequence_memory import SegmentStorage

OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "site", "tests", "golden", "learning_numenta.json"
)


def vec(t):
    return [float(x) for x in t.detach().flatten().tolist()]


def ivec(t):
    return [int(x) for x in t.detach().flatten().tolist()]


def make_net(dt=1.0):
    return Network(behavior={1: TimeResolution(dt=dt)}, dtype=torch.float32, device="cpu")


# ---------------------------------------------------------------------------
# 1. Dendritic segment learning
# ---------------------------------------------------------------------------

def dendritic_segment_scenario():
    """DendriticSegment.learn_on_segments multi-step test with CoNeX defaults.

    Demonstrates crossing the learning threshold with context overlap:
    - Step 0: context overlap = 10 (>= learning_threshold 8) -> matches, reinforces active,
      punishes inactive, and grows new synapses into empty slots.
    - Step 1: context overlap = 5 (< learning_threshold 8) -> does NOT match, no weight changes.
    - Step 2: context overlap = 12 (>= learning_threshold 8) -> matches again, further learning.
    """
    cfg = DendriticSegmentConfig(
        n_synapses_per_segment=20,
        activation_threshold=10,
        learning_threshold=8,
        permanence_increment=0.1,
        permanence_decrement=0.02,
        initial_permanence=0.21,
        connected_permanence=0.2,
        max_new_synapses=5,
    )
    net = make_net()
    ds = DendriticSegment(n_segments=2, n_cells=1, config=cfg)
    neurons = NeuronGroup(net=net, size=1, behavior={10: ds}, tag="ds_cell")
    net.initialize()

    # Initial state for cell 0:
    # Segment 0: 15 initialized synapses to targets 0..14 (perm 0.25) + 5 empty slots (-1, perm 0.0)
    # Segment 1: 15 initialized synapses to targets 15..29 (perm 0.25) + 5 empty slots (-1, perm 0.0)
    s0_targets = list(range(15)) + [-1] * 5
    s0_perms = [0.25] * 15 + [0.0] * 5
    s1_targets = list(range(15, 30)) + [-1] * 5
    s1_perms = [0.25] * 15 + [0.0] * 5

    neurons.segment_targets[0] = torch.tensor(s0_targets, dtype=torch.long)
    neurons.segment_permanences[0] = torch.tensor(s0_perms, dtype=torch.float32)

    neurons.segment_targets[1] = torch.tensor(s1_targets, dtype=torch.long)
    neurons.segment_permanences[1] = torch.tensor(s1_perms, dtype=torch.float32)

    steps_data = []

    # Step 0: overlap 10 (targets 0..9 active) + novel 35..39 active (total 15 active cells)
    pre0 = torch.zeros(40, dtype=torch.bool)
    pre0[0:10] = True
    pre0[35:40] = True

    # Step 1: overlap 5 (targets 0..4 active) -> total active < 8 -> no match
    pre1 = torch.zeros(40, dtype=torch.bool)
    pre1[0:5] = True

    # Step 2: overlap 12 (targets 0..11 active) -> matches
    pre2 = torch.zeros(40, dtype=torch.bool)
    pre2[0:12] = True

    pre_steps = [pre0, pre1, pre2]

    for step_idx, pre in enumerate(pre_steps):
        targets_before = [ivec(neurons.segment_targets[s]) for s in range(2)]
        perms_before = [vec(neurons.segment_permanences[s]) for s in range(2)]

        ds.compute_segment_activation(neurons, pre)
        ds.compute_depolarization(neurons)

        act = vec(neurons.segment_activation)
        active_segs = neurons.segment_active.tolist()
        match_segs = neurons.matching_segments.tolist()
        depol = float(neurons.depolarization[0])

        active_cells = torch.tensor([True])
        winner_cells = torch.tensor([True])

        torch.manual_seed(100 + step_idx)
        ds.learn_on_segments(neurons, active_cells, winner_cells, pre)

        targets_after = [ivec(neurons.segment_targets[s]) for s in range(2)]
        perms_after = [vec(neurons.segment_permanences[s]) for s in range(2)]

        steps_data.append({
            "step": step_idx,
            "presynaptic_activity": [bool(x) for x in pre.tolist()],
            "targets_before": targets_before,
            "permanences_before": perms_before,
            "segment_activation": act,
            "segment_active": active_segs,
            "matching_segments": match_segs,
            "depolarization": depol,
            "targets_after": targets_after,
            "permanences_after": perms_after,
        })

    return {
        "params": {
            "n_cells": 1,
            "n_segments": 2,
            "n_synapses_per_segment": 20,
            "activation_threshold": 10,
            "learning_threshold": 8,
            "permanence_increment": 0.1,
            "permanence_decrement": 0.02,
            "initial_permanence": 0.21,
            "connected_permanence": 0.2,
            "max_new_synapses": 5,
        },
        "steps": steps_data,
    }


def temporal_memory_segments_scenario():
    """TemporalMemory SegmentStorage: adapt_segment, punish_segments, grow_synapses."""
    device = torch.device("cpu")
    dtype = torch.float32

    storage = SegmentStorage(
        n_cells=4,
        max_segments=5,
        max_synapses=10,
        device=device,
        dtype=dtype,
    )

    # 1. adapt_segment
    cell_idx = 0
    seg_idx = 0
    storage.segments[(cell_idx, seg_idx)] = {
        "targets": torch.tensor([1, 2, 3], device=device, dtype=torch.long),
        "permanences": torch.tensor([0.30, 0.15, 0.25], device=device, dtype=dtype),
    }
    storage.segment_counts[cell_idx] = 1

    active_cells = torch.tensor([False, True, False, True], dtype=torch.bool, device=device)
    storage.adapt_segment(cell_idx, seg_idx, active_cells, perm_inc=0.1, perm_dec=0.02)
    adapted_perms = vec(storage.segments[(cell_idx, seg_idx)]["permanences"])

    # 2. punish_segments
    # Segment caused false prediction: active connected synapses are decremented
    # cell 1, segment 0
    storage.segments[(1, 0)] = {
        "targets": torch.tensor([1, 3], device=device, dtype=torch.long),
        "permanences": torch.tensor([0.40, 0.35], device=device, dtype=dtype),
    }
    storage.segment_counts[1] = 1
    storage.punish_segments(cell_idx=1, active_cells=active_cells, decrement=0.05, threshold=1, connected_perm=0.2)
    punished_perms = vec(storage.segments[(1, 0)]["permanences"])

    # 3. grow_synapses
    # cell 0, segment 0: currently targets [1, 2, 3]. Winner cells has [0, 2] -> only 0 is candidate
    winner_cells = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
    torch.manual_seed(77)
    storage.grow_synapses(
        cell_idx=0,
        segment_idx=0,
        winner_cells=winner_cells,
        initial_perm=0.21,
        max_new=2,
        max_synapses=10,
    )
    grown_targets = ivec(storage.segments[(0, 0)]["targets"])
    grown_perms = vec(storage.segments[(0, 0)]["permanences"])

    return {
        "adapt": {
            "cell_idx": 0,
            "segment_idx": 0,
            "initial_targets": [1, 2, 3],
            "initial_perms": [0.30, 0.15, 0.25],
            "active_cells": [False, True, False, True],
            "perm_inc": 0.1,
            "perm_dec": 0.02,
            "expected_perms": adapted_perms,
        },
        "punish": {
            "cell_idx": 1,
            "segment_idx": 0,
            "initial_targets": [1, 3],
            "initial_perms": [0.40, 0.35],
            "active_cells": [False, True, False, True],
            "decrement": 0.05,
            "threshold": 1,
            "connected_perm": 0.2,
            "expected_perms": punished_perms,
        },
        "grow": {
            "cell_idx": 0,
            "segment_idx": 0,
            "winner_cells": [True, False, True, False],
            "initial_perm": 0.21,
            "max_new": 2,
            "max_synapses": 10,
            "expected_targets": grown_targets,
            "expected_perms": grown_perms,
        },
    }


# ---------------------------------------------------------------------------
# 2. PredictiveSynapseLearning
# ---------------------------------------------------------------------------

def predictive_synapse_scenario():
    """PredictiveSynapseLearning for error_driven, hebbian, combined."""
    dt = 1.0
    cfg = PredictiveCodingConfig(learning_rate=0.05)

    def run_type(ltype):
        net = make_net(dt=dt)
        src = NeuronGroup(net=net, size=3, tag="src")
        dst = NeuronGroup(net=net, size=2, tag="dst")
        syn = SynapseGroup(
            net=net, src=src, dst=dst, tag="syn",
            behavior={
                10: PredictiveSynapseInit(config=cfg, connection_type="feedforward", init_scale=0.1),
                20: PredictiveSynapseLearning(config=cfg, learning_type=ltype),
            },
        )
        net.initialize()
        # Fixed initial weights and eligibility
        syn.weights = torch.tensor([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.1]], dtype=torch.float32)
        syn.eligibility = torch.zeros(3, 2, dtype=torch.float32)

        src.activity = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        dst.prediction_error = torch.tensor([-0.2, 0.4], dtype=torch.float32)
        dst.activity = torch.tensor([0.8, 0.1], dtype=torch.float32)

        learning = syn.behavior[20]
        w_trace = []
        el_trace = []
        for _ in range(4):
            learning.forward(syn)
            w_trace.append(vec(syn.weights))
            el_trace.append(vec(syn.eligibility))

        return {
            "weights_trace": w_trace,
            "eligibility_trace": el_trace,
        }

    return {
        "params": {
            "learning_rate": 0.05,
            "dt": 1.0,
            "src_size": 3,
            "dst_size": 2,
            "initial_weights": [0.2, -0.1, 0.5, 0.3, -0.4, 0.1],
            "src_activity": [1.0, 0.5, 0.0],
            "dst_error": [-0.2, 0.4],
            "dst_activity": [0.8, 0.1],
        },
        "error_driven": run_type("error_driven"),
        "hebbian": run_type("hebbian"),
        "combined": run_type("combined"),
    }


# ---------------------------------------------------------------------------
# 3. Precision rule _learn_precision
# ---------------------------------------------------------------------------

def precision_rule_scenario():
    """PrecisionWeighting._learn_precision direct test."""
    net = make_net()
    cfg = PredictiveCodingConfig(
        precision_learning_rate=0.02,
        min_precision=0.05,
        max_precision=50.0,
    )
    ng = NeuronGroup(
        net=net, size=4,
        behavior={10: PrecisionWeighting(config=cfg, learnable=True, attention_modulated=False)},
    )
    net.initialize()

    ng.sensory_precision = torch.tensor([1.0, 2.0, 0.5, 4.0], dtype=torch.float32)
    ng.signed_error = torch.tensor([0.1, -0.5, 1.2, 0.0], dtype=torch.float32)

    pw = ng.behavior[10]
    pw._learn_precision(ng)

    return {
        "params": {
            "precision_learning_rate": 0.02,
            "min_precision": 0.05,
            "max_precision": 50.0,
            "dt": 1.0,
            "initial_sensory_precision": [1.0, 2.0, 0.5, 4.0],
            "signed_error": [0.1, -0.5, 1.2, 0.0],
        },
        "precision_grad": vec(ng.precision_grad),
        "updated_sensory_precision": vec(ng.sensory_precision),
    }


# ---------------------------------------------------------------------------
# 4. Predictive coding full learning network (heatmap + precision + error)
# ---------------------------------------------------------------------------

def predictive_network_scenario():
    """PredictiveCodingLearning in full network: weights heatmap, error, precision."""
    size = 8
    steps = 20
    net = make_net()
    cfg = PredictiveCodingConfig(
        learning_rate=0.05,
        precision_tau=20.0,
        precision_learning_rate=0.01,
        min_precision=0.01,
        max_precision=100.0,
        error_nonlinearity="linear",
        use_sparse_errors=False,
    )
    behaviors = {
        230: TopDownPrediction(config=cfg),
        234: ErrorUnit(config=cfg, precision_weighted=False),
        236: PrecisionWeighting(config=cfg, learnable=True, attention_modulated=False),
        238: FreeEnergyMinimization(config=cfg, track_components=True),
        350: PredictiveCodingLearning(config=cfg, use_hebbian=False),
    }
    neurons = NeuronGroup(net=net, size=size, behavior=behaviors, tag="pc_net")
    net.initialize()

    # Fixed reproducible initial weights and inputs
    neurons.top_down_weights = torch.eye(size, dtype=torch.float32) * 0.5
    sensory = torch.linspace(0.1, 0.9, size)
    higher = torch.linspace(0.9, 0.1, size)
    neurons.activity = sensory
    neurons.top_down_input = higher

    order = [neurons.behavior[k] for k in sorted(neurons.behavior)]

    trace = []
    for step in range(steps):
        for b in order:
            b.forward(neurons)

        trace.append({
            "step": step,
            "mean_abs_error": float(neurons.error_magnitude.mean()),
            "mean_precision": float(neurons.precision.mean()),
            "free_energy": float(neurons.free_energy),
            "prediction": vec(neurons.prediction),
            "signed_error": vec(neurons.signed_error),
            "precision": vec(neurons.precision),
            "top_down_weights": vec(neurons.top_down_weights),
        })

    return {
        "size": size,
        "steps": steps,
        "params": {
            "learning_rate": 0.05,
            "precision_tau": 20.0,
            "precision_learning_rate": 0.01,
            "min_precision": 0.01,
            "max_precision": 100.0,
            "error_nonlinearity": "linear",
            "use_sparse_errors": False,
        },
        "sensory": vec(sensory),
        "higher": vec(higher),
        "trace": trace,
    }


def main():
    data = {
        "dendritic_segments": dendritic_segment_scenario(),
        "temporal_memory_segments": temporal_memory_segments_scenario(),
        "predictive_synapse": predictive_synapse_scenario(),
        "precision_rule": precision_rule_scenario(),
        "predictive_network": predictive_network_scenario(),
    }
    out = os.path.normpath(OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
