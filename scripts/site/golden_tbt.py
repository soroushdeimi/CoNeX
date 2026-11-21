"""Golden reference numbers for the tbt area of the site.

Builds small CoNeX networks with the real classes, runs them, and writes
site/tests/golden/tbt.json. The JS models in site/assets/js/models/tbt.js
must reproduce these numbers.

Run:
    /Users/soroushdeimi/Desktop/myprojects/.conex-site-venv/bin/python scripts/site/golden_tbt.py
"""

import json
import math
import os

import torch
from pymonntorch import Network, NeuronGroup

from conex import (
    ActiveDendriteComputation,
    ColumnVoting,
    ConsensusNetwork,
    ContextualPrediction,
    DendriticSegment,
    DendriticSegmentConfig,
    DisplacementCellModule,
    ErrorUnit,
    FreeEnergyMinimization,
    GridCellModule,
    PrecisionWeighting,
    PredictiveCodingConfig,
    PredictiveCodingLearning,
    SDROverlap,
    TimeResolution,
    TopDownPrediction,
)
from conex.nn.structure.numenta.predictive_hierarchy import PredictiveCorticalColumn

OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "site", "tests", "golden", "tbt.json"
)


def vec(t):
    return [float(x) for x in t.detach().flatten().tolist()]


def make_net(dt=1.0):
    return Network(behavior={1: TimeResolution(dt=dt)}, dtype=torch.float32, device="cpu")


def grid_scenario():
    """GridCellModule: scales, orientations, phases, activations, path integration."""
    torch.manual_seed(1234)
    n_modules, cells_per_module = 4, 25
    net = make_net()
    module = GridCellModule(
        n_modules=n_modules,
        cells_per_module=cells_per_module,
        scale_range=(0.5, 2.0),
        orientation_range=(0.0, math.pi / 3),
    )
    neurons = NeuronGroup(
        net=net, size=n_modules * cells_per_module, behavior={100: module}, tag="grid"
    )
    net.initialize()

    positions = [[0.0, 0.0], [1.25, -0.75], [2.0, 3.0]]
    activations = [
        vec(module.encode_position(neurons, torch.tensor(p))) for p in positions
    ]

    # Path integration: two velocity steps of dt = 0.5 s
    module.path_integrate(neurons, torch.tensor([1.0, 0.0]), dt=0.5)
    module.path_integrate(neurons, torch.tensor([0.0, 2.0]), dt=0.5)
    position = vec(neurons.grid_position)

    # Lattice periodicity of module 0: translation 2*lambda/sqrt(3) at theta+30 deg
    scale = module.scales[0].item()
    angle = module.orientations[0].item() + math.pi / 6
    step = torch.tensor(
        [2 * scale / math.sqrt(3) * math.cos(angle), 2 * scale / math.sqrt(3) * math.sin(angle)]
    )
    origin = vec(module.encode_position(neurons, torch.zeros(2))[:cells_per_module])
    shifted = vec(module.encode_position(neurons, step)[:cells_per_module])

    return {
        "n_modules": n_modules,
        "cells_per_module": cells_per_module,
        "scale_range": [0.5, 2.0],
        "orientation_range": [0.0, math.pi / 3],
        "seed": 1234,
        "scales": vec(module.scales),
        "orientations": vec(module.orientations),
        "phases": [vec(row) for row in neurons.grid_phase],
        "wave_vectors": [vec(row) for row in module.wave_vectors],
        "positions": positions,
        "activations": activations,
        "velocity_steps": [[1.0, 0.0], [0.0, 2.0]],
        "dt": 0.5,
        "position_after_integration": position,
        "periodicity": {"origin": origin, "shifted": shifted, "step": vec(step)},
    }


def displacement_scenario():
    """DisplacementCellModule: preferred displacements, encode, decode."""
    net = make_net()
    module = DisplacementCellModule(n_cells=100, max_displacement=10.0, sigma=0.5)
    neurons = NeuronGroup(net=net, size=100, behavior={100: module}, tag="disp")
    net.initialize()

    from_loc = torch.tensor([1.0, 2.0])
    to_loc = torch.tensor([4.0, 5.0])
    activation = vec(module.compute_displacement(neurons, from_loc, to_loc))
    decoded = vec(module.decode_displacement(neurons, torch.tensor(activation)))

    # A second pair of locations
    from2 = torch.tensor([0.0, 0.0])
    to2 = torch.tensor([0.5, -1.5])
    activation2 = vec(module.compute_displacement(neurons, from2, to2))
    decoded2 = vec(module.decode_displacement(neurons, torch.tensor(activation2)))

    return {
        "n_cells": 100,
        "max_displacement": 10.0,
        "sigma": 0.5,
        "preferred_dx": vec(module.preferred_dx),
        "preferred_dy": vec(module.preferred_dy),
        "pair1": {"from": [1.0, 2.0], "to": [4.0, 5.0],
                  "activation": activation, "decoded": decoded},
        "pair2": {"from": [0.0, 0.0], "to": [0.5, -1.5],
                  "activation": activation2, "decoded": decoded2},
    }


def dendrites_scenario():
    """DendriticSegment + ActiveDendriteComputation + ContextualPrediction."""
    n_cells, n_segments, n_synapses = 8, 2, 4
    config = DendriticSegmentConfig(
        n_synapses_per_segment=n_synapses,
        activation_threshold=2,
        learning_threshold=1,
        connected_permanence=0.5,
        initial_permanence=0.6,
        permanence_increment=0.1,
        permanence_decrement=0.05,
        max_new_synapses=2,
    )
    net = make_net()
    module = DendriticSegment(n_segments=n_segments, n_cells=n_cells, config=config)
    neurons = NeuronGroup(net=net, size=n_cells, behavior={100: module}, tag="seg")
    net.initialize()

    # Wire segment 0 (cell 0) to presynaptic cells 1, 2, 3 with permanence 0.9
    for slot, target in enumerate([1, 2, 3]):
        neurons.segment_targets[0, slot] = target
        neurons.segment_permanences[0, slot] = 0.9

    active = torch.zeros(n_cells, dtype=torch.bool)
    active[[1, 2, 3]] = True
    module.compute_segment_activation(neurons, active)
    module.compute_depolarization(neurons)
    all_active = {
        "segment_activation": vec(neurons.segment_activation),
        "segment_active": [bool(x) for x in neurons.segment_active],
        "matching": [bool(x) for x in neurons.matching_segments],
        "depolarization": vec(neurons.depolarization),
    }

    # A second context where only one of the three cells is active
    active_one = torch.zeros(n_cells, dtype=torch.bool)
    active_one[2] = True
    module.compute_segment_activation(neurons, active_one)
    module.compute_depolarization(neurons)
    one_active = {
        "segment_activation": vec(neurons.segment_activation),
        "segment_active": [bool(x) for x in neurons.segment_active],
        "matching": [bool(x) for x in neurons.matching_segments],
        "depolarization": vec(neurons.depolarization),
    }

    # ActiveDendriteComputation: NMDA nonlinearity and integration
    net2 = make_net()
    dend = ActiveDendriteComputation(nmda_threshold=0.5, nmda_gain=2.0)
    dend_neurons = NeuronGroup(net=net2, size=n_cells, behavior={100: dend}, tag="dend")
    net2.initialize()

    nmda_low = vec(dend.nmda_nonlinearity(torch.tensor([0.1])))
    nmda_high = vec(dend.nmda_nonlinearity(torch.tensor([1.0])))

    for _ in range(20):
        current = dend.compute_dendritic_integration(
            dend_neurons,
            proximal_input=torch.ones(n_cells),
            distal_input=torch.full((n_cells,), 3.0),
        )
    integration = {
        "steps": 20,
        "proximal_input": 1.0,
        "distal_input": 3.0,
        "proximal_potential": vec(dend_neurons.proximal_potential),
        "distal_potential": vec(dend_neurons.distal_potential),
        "dendritic_spike": [bool(x) for x in dend_neurons.dendritic_spike],
        "dendritic_calcium": vec(dend_neurons.dendritic_calcium),
        "total_current": vec(current),
    }

    # ContextualPrediction
    net3 = make_net()
    ctx = ContextualPrediction(
        depolarization_threshold=0.5, prediction_window=3, burst_threshold=0.3
    )
    ctx_neurons = NeuronGroup(net=net3, size=n_cells, behavior={100: ctx}, tag="ctx")
    net3.initialize()

    drive = torch.zeros(n_cells)
    drive[[0, 1]] = 0.9
    ctx.update_predictions(ctx_neurons, drive)
    activation, bursting = ctx.process_activation(ctx_neurons, torch.ones(n_cells))

    return {
        "segment": {
            "n_cells": n_cells,
            "n_segments": n_segments,
            "n_synapses": n_synapses,
            "activation_threshold": 2,
            "learning_threshold": 1,
            "connected_permanence": 0.5,
            "targets": [1, 2, 3],
            "permanence": 0.9,
            "all_active": all_active,
            "one_active": one_active,
        },
        "nmda": {"low": nmda_low, "high": nmda_high},
        "integration": integration,
        "context": {
            "depolarization_threshold": 0.5,
            "prediction_window": 3,
            "burst_threshold": 0.3,
            "drive": vec(drive),
            "predictive": [bool(x) for x in ctx_neurons.predictive],
            "activation": vec(activation),
            "bursting": [bool(x) for x in bursting],
        },
    }


def voting_scenario():
    """ColumnVoting + ConsensusNetwork: three columns, one true object."""
    torch.manual_seed(7)
    n_objects, rep_size, sparsity = 6, 64, 0.25
    n_active = max(1, int(rep_size * sparsity))  # 16

    # Object memories: fixed random SDRs
    memories = torch.zeros(n_objects, rep_size)
    for o in range(n_objects):
        idx = torch.randperm(rep_size)[:n_active]
        memories[o, idx] = 1.0

    # True object is 2. Each column's local representation is the true SDR
    # with a few bits flipped (ambiguous candidates).
    true_object = 2
    flips = [0, 2, 4]
    local_reps = []
    for c in range(3):
        rep = memories[true_object].clone()
        on = rep.nonzero(as_tuple=True)[0]
        off = (rep == 0).nonzero(as_tuple=True)[0]
        rep[on[: flips[c]]] = 0.0
        rep[off[: flips[c]]] = 1.0
        local_reps.append(rep)

    net = make_net()
    module = ColumnVoting(
        n_objects=n_objects,
        representation_size=rep_size,
        sparsity=sparsity,
        voting_threshold=0.5,
        decay_rate=0.1,
        min_votes_for_consensus=3,
    )
    columns = []
    for c in range(3):
        col = NeuronGroup(net=net, size=rep_size, behavior={100: module}, tag=f"col{c}")
        columns.append(col)
    net.initialize()

    consensus = ConsensusNetwork(
        convergence_threshold=0.8, max_iterations=20, broadcast_interval=1
    )
    consensus.initialize(net)

    steps = []
    for step in range(10):
        for c, col in enumerate(columns):
            votes = module.cast_vote(col, local_reps[c], memories)
            module.receive_votes(col, votes)
        consensus.broadcast_votes(net, columns)
        for col in columns:
            module.forward(col)  # decay
        reached = [bool(module.check_consensus(col, 3)) for col in columns]
        net_consensus = consensus.check_network_consensus(net, columns)
        consensus.forward(net)
        steps.append(
            {
                "step": step,
                "accumulators": [vec(col.vote_accumulator) for col in columns],
                "consensus": reached,
                "consensus_objects": [col.consensus_object for col in columns],
                "network_consensus": bool(net_consensus),
            }
        )

    # SDR overlap metrics on the memories
    overlap = SDROverlap.compute_overlap(memories[0], memories[1])
    ratio = SDROverlap.compute_overlap_ratio(memories[0], memories[1])
    jaccard = SDROverlap.compute_jaccard(memories[0], memories[1])

    return {
        "seed": 7,
        "n_objects": n_objects,
        "representation_size": rep_size,
        "sparsity": sparsity,
        "voting_threshold": 0.5,
        "decay_rate": 0.1,
        "min_votes_for_consensus": 3,
        "convergence_threshold": 0.8,
        "true_object": true_object,
        "memories": [vec(row) for row in memories],
        "local_reps": [vec(r) for r in local_reps],
        "steps": steps,
        "sdr_metrics": {"overlap": overlap, "ratio": ratio, "jaccard": jaccard},
    }


def predictive_scenario():
    """Predictive coding level, wired as in Example/numenta/predictive_coding.py.

    TopDownPrediction -> ErrorUnit -> PrecisionWeighting
        -> FreeEnergyMinimization -> PredictiveCodingLearning
    """
    torch.manual_seed(7)
    size, steps = 32, 40

    config = PredictiveCodingConfig(
        error_nonlinearity="linear",
        use_sparse_errors=False,
        learning_rate=0.05,
        precision_tau=20.0,
        precision_learning_rate=0.01,
    )
    net = make_net()
    behaviors = {
        230: TopDownPrediction(config=config),
        234: ErrorUnit(config=config, precision_weighted=False),
        236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
        238: FreeEnergyMinimization(config=config, track_components=True),
        350: PredictiveCodingLearning(config=config, use_hebbian=False),
    }
    neurons = NeuronGroup(net=net, size=size, behavior=behaviors, tag="level")
    net.initialize()

    sensory = torch.rand(size)
    higher_level = torch.rand(size)
    neurons.activity = sensory
    neurons.top_down_input = higher_level

    order = [neurons.behavior[k] for k in sorted(neurons.behavior)]

    def advance():
        for behavior in order:
            behavior.forward(neurons)

    trace = []
    for i in range(steps):
        advance()
        trace.append(
            {
                "step": i,
                "mean_abs_error": float(neurons.error_magnitude.mean()),
                "mean_precision": float(neurons.precision.mean()),
                "free_energy": float(neurons.free_energy),
                "accuracy_term": float(neurons.accuracy_term),
                "complexity_term": float(neurons.complexity_term),
            }
        )

    # Perturb the sensory input: the error jumps again
    neurons.activity = sensory + torch.randn(size) * 0.4
    perturb = []
    for i in range(6):
        advance()
        perturb.append(
            {
                "step": i,
                "mean_abs_error": float(neurons.error_magnitude.mean()),
                "free_energy": float(neurons.free_energy),
            }
        )

    return {
        "seed": 7,
        "size": size,
        "steps": steps,
        "config": {
            "prediction_tau": config.prediction_tau,
            "error_tau": config.error_tau,
            "precision_tau": config.precision_tau,
            "learning_rate": config.learning_rate,
            "precision_learning_rate": config.precision_learning_rate,
            "min_precision": config.min_precision,
            "max_precision": config.max_precision,
            "error_nonlinearity": config.error_nonlinearity,
            "use_sparse_errors": config.use_sparse_errors,
            "error_threshold": config.error_threshold,
            "temporal_smoothing": config.temporal_smoothing,
        },
        "sensory": vec(sensory),
        "higher_level": vec(higher_level),
        "trace": trace,
        "perturbation": perturb,
    }


def hierarchy_scenario():
    """HierarchicalPredictiveCoding: two levels, fixed weights, one forward pass."""
    torch.manual_seed(42)
    level_sizes = [8, 4]
    ff = torch.randn(level_sizes[0], level_sizes[1]) * 0.1
    fb = torch.randn(level_sizes[1], level_sizes[0]) * 0.1
    sensory = torch.rand(level_sizes[0])

    from conex.behaviors.neurons.numenta.predictive_coding import (
        HierarchicalPredictiveCoding,
    )

    net = make_net()
    module = HierarchicalPredictiveCoding(n_levels=2, level_sizes=level_sizes)
    neurons = NeuronGroup(net=net, size=level_sizes[0], behavior={100: module}, tag="hier")
    net.initialize()
    # Overwrite the random weights with our fixed ones for reproducibility.
    neurons.feedforward_weights[0] = ff
    neurons.feedback_weights[0] = fb
    neurons.activity = sensory

    module.forward(neurons)

    return {
        "seed": 42,
        "level_sizes": level_sizes,
        "feedforward_weights": vec(ff),
        "feedback_weights": vec(fb),
        "sensory": vec(sensory),
        "level_predictions": [vec(p) for p in neurons.level_predictions],
        "level_errors": [vec(e) for e in neurons.level_errors],
    }


def structure_scenario():
    """PredictiveCorticalColumn: layer sizes and the connections it builds."""
    torch.manual_seed(11)
    net = make_net()
    column = PredictiveCorticalColumn(net, column_size=100, name="col")
    net.initialize()

    return {
        "column_size": 100,
        "layer_sizes": {k: int(v) for k, v in column.layer_sizes.items()},
        "connections": sorted(column.synapses.keys()),
        "population_sizes": {k: int(v.size) for k, v in column.populations.items()},
    }


def main():
    data = {
        "grid": grid_scenario(),
        "displacement": displacement_scenario(),
        "dendrites": dendrites_scenario(),
        "voting": voting_scenario(),
        "predictive": predictive_scenario(),
        "hierarchy": hierarchy_scenario(),
        "structure": structure_scenario(),
    }
    out = os.path.normpath(OUT)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
