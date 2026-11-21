"""Golden numbers for the HTM site plates.

Runs the real CoNeX classes (SDR, SDREncoder, SDRClassifier, UnionSDR,
SpatialPooler, TemporalMemory) on small fixed inputs and writes
site/tests/golden/htm.json. The JS models in site/assets/js/models/htm.js
must reproduce these numbers.

Determinism notes:
- SDR / encoder / classifier / union scenarios are fully deterministic.
- SpatialPooler: the random potential connections and initial permanences
  are overwritten with fixed values after initialize(), so only the
  compute/learn/duty/boost logic is pinned.
- TemporalMemory: at most 2 active columns per step and
  max_new_synapses=5, so every synapse-growth candidate set is selected
  whole and torch.randperm only shuffles the order, never the set.
- The false-match probabilities are the exact hypergeometric tail
  P(B >= theta) = sum_b C(w,b) C(n-w, w-b) / C(n, w)
  (Ahmad and Hawkins 2016, eq. 3; same formula as
  neurosush htm/sdr.py match_probability).
"""

import json
import math
import os
import sys
from fractions import Fraction

import torch
from pymonntorch import Network, NeuronGroup

from conex import (
    SDR,
    SDRClassifier,
    SDREncoder,
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
    TimeResolution,
    UnionSDR,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "site", "tests", "golden", "htm.json")


def make_group(size, behavior, tag="test", net=None):
    owns_network = net is None
    if owns_network:
        net = Network(
            behavior={1: TimeResolution(dt=1.0)},
            dtype=torch.float32,
            device="cpu",
        )
    neurons = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    if owns_network:
        net.initialize()
    return neurons


def sdr_golden():
    size = 64
    a = SDR(size=size, indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
    b = SDR(size=size, indices=torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    c = SDR(size=size)  # empty
    union = a.union(b)
    inter = a.intersection(b)
    diff = a.difference(b)
    return {
        "size": size,
        "a_indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "b_indices": [4, 5, 6, 7, 8, 9, 10, 11],
        "n_active": [a.n_active, b.n_active, c.n_active],
        "sparsity": [a.sparsity, b.sparsity, c.sparsity],
        "overlap_ab": a.overlap(b),
        "overlap_ac": a.overlap(c),
        "overlap_score_ab": a.overlap_score(b),
        "overlap_score_ac": a.overlap_score(c),
        "jaccard_ab": a.jaccard(b),
        "union_n_active": union.n_active,
        "union_indices": union.indices.tolist(),
        "intersection_indices": inter.indices.tolist(),
        "difference_indices": diff.indices.tolist(),
        "match_threshold_ab_4": a.match_threshold(b, 4),
        "match_threshold_ab_5": a.match_threshold(b, 5),
        "config_n_active": [
            max(1, int(2048 * 0.02)),   # SDRConfig(2048, 0.02)
            max(1, int(100 * 0.001)),   # SDRConfig(100, 0.001) -> max(1, 0)
        ],
    }


def false_match_golden():
    cases = [(64, 8, 4), (64, 8, 5), (64, 8, 6), (2048, 40, 10), (2048, 40, 15), (2048, 40, 20)]
    out = []
    for n, w, theta in cases:
        hits = sum(
            math.comb(w, b) * math.comb(n - w, w - b)
            for b in range(theta, min(w, w) + 1)
            if w - b <= n - w
        )
        p = float(Fraction(hits, math.comb(n, w)))
        out.append({"n": n, "w": w, "theta": theta, "p": p, "expected_overlap": w * w / n})
    return out


def encoder_golden():
    size, n_active = 64, 8
    enc = make_group(size, SDREncoder(size=size, n_active=n_active, encoder_type="scalar",
                                      min_val=0.0, max_val=1.0), tag="enc")
    e = enc.sdr_encoder
    values = [0.0, 0.5, 0.52, 1.0, -5.0, 5.0]
    codes = {str(v): e.encode_scalar(v) for v in values}
    return {
        "size": size,
        "n_active": n_active,
        "min_val": 0.0,
        "max_val": 1.0,
        "codes": {k: c.indices.tolist() for k, c in codes.items()},
        "overlap_0.5_0.52": codes["0.5"].overlap(codes["0.52"]),
        "overlap_0.5_0.0": codes["0.5"].overlap(codes["0.0"]),
        "overlap_0.52_1.0": codes["0.52"].overlap(codes["1.0"]),
        "clamp_equal": codes["-5.0"] == codes["0.0"] and codes["5.0"] == codes["1.0"],
    }


def classifier_golden():
    size, n_classes = 64, 3
    clf = make_group(size, SDRClassifier(n_classes=n_classes, sdr_size=size), tag="clf")
    c = clf.sdr_classifier
    patterns = {
        0: SDR(size=size, indices=torch.arange(0, 8)),
        1: SDR(size=size, indices=torch.arange(20, 28)),
        2: SDR(size=size, indices=torch.arange(40, 48)),
    }
    for label, pat in patterns.items():
        for _ in range(5):
            c.learn(clf, pat, label)
    pred0, scores0 = c.infer(clf, patterns[0])
    pred1, scores1 = c.infer(clf, patterns[1])
    return {
        "n_classes": n_classes,
        "sdr_size": size,
        "learning_rate": 0.01,
        "n_examples_per_class": 5,
        "predict_0": pred0,
        "scores_0": scores0.tolist(),
        "predict_1": pred1,
        "scores_1": scores1.tolist(),
        "class_counts": clf.class_counts.tolist(),
    }


def sp_golden():
    input_size, n_columns = 16, 8
    # potential_pct=1.0 -> n_potential = input_size, so every column can
    # connect to every input and the potential map is the identity.
    config = SpatialPoolerConfig(
        input_size=input_size,
        n_columns=n_columns,
        potential_pct=1.0,
        local_area_density=0.25,
        syn_perm_connected=0.1,
        syn_perm_active_inc=0.06,
        syn_perm_inactive_dec=0.01,
        duty_cycle_period=4,
        boost_strength=0.0,
        seed=7,
    )
    pooler = make_group(n_columns, SpatialPooler(config=config), tag="sp")
    sp = pooler.spatial_pooler

    # Overwrite the random init with fixed values so the golden pins the
    # compute/learn/duty/boost logic, not torch's random init. Column c is
    # connected to inputs 0..(7+c), so overlaps are distinct and the top-k
    # winners are unambiguous (no tie-breaking to mirror).
    pooler.sp_potential_inputs = torch.arange(input_size).repeat(n_columns).view(n_columns, input_size)
    perm = torch.zeros(n_columns, input_size)
    for c in range(n_columns):
        for i in range(input_size):
            perm[c, i] = 0.1 if i < (8 + c) else 0.05
    pooler.sp_permanences = perm
    assert sp.n_potential == input_size

    # Input with bits 8..15 active. Column c has (8+c) connected inputs, of
    # which the active ones are inputs 8..(7+c): count = c. So overlaps are
    # 0,1,2,...,7 and the top-2 winners are columns 6 and 7.
    y = torch.zeros(input_size)
    y[8:16] = 1.0

    active1 = sp.compute(pooler, y, learn=True)
    perm_after_1 = pooler.sp_permanences[6].tolist()
    duty_after_1 = [pooler.sp_overlap_duty[6].item(), pooler.sp_active_duty[6].item()]

    # Same input, no learning: winners unchanged.
    active2 = sp.compute(pooler, y, learn=False)

    # Same input again with learning: permanences drift, winners unchanged.
    active3 = sp.compute(pooler, y, learn=True)
    perm_after_3 = pooler.sp_permanences[6].tolist()

    # Empty input: overlap 0 everywhere, nothing active.
    active_empty = sp.compute(pooler, torch.zeros(input_size), learn=False)

    # Boosting: boost_strength > 0, run a few steps, read the factors.
    config_b = SpatialPoolerConfig(
        input_size=input_size,
        n_columns=n_columns,
        potential_pct=1.0,
        local_area_density=0.25,
        syn_perm_connected=0.1,
        duty_cycle_period=4,
        boost_strength=1.0,
        seed=7,
    )
    pooler_b = make_group(n_columns, SpatialPooler(config=config_b), tag="spb")
    spb = pooler_b.spatial_pooler
    pooler_b.sp_potential_inputs = torch.arange(input_size).repeat(n_columns).view(n_columns, input_size)
    perm_b = torch.zeros(n_columns, input_size)
    for c in range(n_columns):
        for i in range(input_size):
            perm_b[c, i] = 0.1 if i < (8 + c) else 0.05
    pooler_b.sp_permanences = perm_b
    for _ in range(3):
        spb.compute(pooler_b, y, learn=True)
    boost = pooler_b.sp_boost_factors.tolist()

    return {
        "input_size": input_size,
        "n_columns": n_columns,
        "potential_pct": 1.0,
        "local_area_density": 0.25,
        "syn_perm_connected": 0.1,
        "syn_perm_active_inc": 0.06,
        "syn_perm_inactive_dec": 0.01,
        "duty_cycle_period": 4,
        "n_active": sp.n_active,
        "n_potential": sp.n_potential,
        "connected_pattern": "col c connected to inputs 0..(7+c), perm 0.1; else 0.05",
        "input_active_bits": [8, 9, 10, 11, 12, 13, 14, 15],
        "active_step1": active1.nonzero(as_tuple=True)[0].tolist(),
        "perm_col6_after_step1": perm_after_1,
        "duty_col6_after_step1": duty_after_1,
        "active_step2": active2.nonzero(as_tuple=True)[0].tolist(),
        "active_step3": active3.nonzero(as_tuple=True)[0].tolist(),
        "perm_col6_after_step3": perm_after_3,
        "active_empty": active_empty.nonzero(as_tuple=True)[0].tolist(),
        "boost_factors_after_3": boost,
    }


def tm_golden():
    # Six 3-column symbols over 18 columns. Each symbol activates 3 columns,
    # so a segment grown from one symbol's winners gets 3 synapses, which
    # reaches activation_threshold=3. max_new_synapses=5 >= 3, so every
    # candidate set is selected whole and growth is deterministic.
    n_columns, cells = 18, 4
    config = TemporalMemoryConfig(
        n_columns=n_columns,
        cells_per_column=cells,
        activation_threshold=3,
        learning_threshold=2,
        max_new_synapses=5,
        permanence_increment=0.12,
        permanence_decrement=0.04,
    )
    memory = make_group(n_columns * cells, TemporalMemory(config=config), tag="tm")
    tm = memory.temporal_memory

    def cols(*idx):
        c = torch.zeros(n_columns, dtype=torch.bool)
        c[list(idx)] = True
        return c

    A = cols(0, 1, 2)
    X = cols(3, 4, 5)
    B = cols(6, 7, 8)
    C = cols(9, 10, 11)
    D = cols(12, 13, 14)
    Y = cols(15, 16, 17)
    seq1 = [A, B, C, D]
    seq2 = [X, B, C, Y]
    for _ in range(8):
        for step in seq1 + seq2:
            tm.compute(memory, step, learn=True)

    # Replay without learning: record active/winner/predictive/bursting.
    replay = []
    for step in seq1 + seq2:
        tm.compute(memory, step, learn=False)
        replay.append({
            "active_cells": memory.tm_active_cells.nonzero(as_tuple=True)[0].tolist(),
            "winner_cells": memory.tm_winner_cells.nonzero(as_tuple=True)[0].tolist(),
            "predictive_cells": memory.tm_predictive_cells.nonzero(as_tuple=True)[0].tolist(),
            "bursting_columns": memory.tm_bursting_columns.nonzero(as_tuple=True)[0].tolist(),
            "anomaly": memory.tm_anomaly_score.item(),
        })

    # Segment counts per cell after learning.
    seg_counts = [memory.tm_segments.count_segments(c) for c in range(n_columns * cells)]

    # A fresh memory: novel input bursts, anomaly 1.0, one winner per column.
    fresh = make_group(n_columns * cells, TemporalMemory(config=config), tag="tm2")
    fresh.temporal_memory.compute(fresh, A, learn=False)
    return {
        "n_columns": n_columns,
        "cells_per_column": cells,
        "activation_threshold": 3,
        "learning_threshold": 2,
        "max_new_synapses": 5,
        "permanence_increment": 0.12,
        "permanence_decrement": 0.04,
        "symbols": {
            "A": [0, 1, 2], "X": [3, 4, 5], "B": [6, 7, 8],
            "C": [9, 10, 11], "D": [12, 13, 14], "Y": [15, 16, 17],
        },
        "segment_counts": seg_counts,
        "replay": replay,
        "fresh": {
            "bursting_columns": fresh.tm_bursting_columns.nonzero(as_tuple=True)[0].tolist(),
            "winner_cells": fresh.tm_winner_cells.nonzero(as_tuple=True)[0].tolist(),
            "anomaly": fresh.tm_anomaly_score.item(),
        },
    }


def union_golden():
    size = 64
    a = SDR(size=size, indices=torch.arange(0, 8))
    b = SDR(size=size, indices=torch.arange(4, 12))
    c = SDR(size=size, indices=torch.arange(60, 68))
    u = UnionSDR(size=size)
    u.add(a)
    u.add(b)
    u.add(c)
    n3 = u.n_active
    sp3 = u.sparsity
    contains_a = u.contains(a, 0.5)
    u.remove(a)
    return {
        "size": size,
        "n_active_after_3": n3,
        "sparsity_after_3": sp3,
        "n_members_after_3": u.n_members + 1,
        "contains_a": contains_a,
        "n_active_after_remove_a": u.n_active,
        "union_indices_after_remove_a": u.to_sdr().indices.tolist(),
    }


def main():
    torch.manual_seed(0)
    golden = {
        "sdr": sdr_golden(),
        "false_match": false_match_golden(),
        "encoder": encoder_golden(),
        "classifier": classifier_golden(),
        "spatial_pooler": sp_golden(),
        "temporal_memory": tm_golden(),
        "union": union_golden(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
