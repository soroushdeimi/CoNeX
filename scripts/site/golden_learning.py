"""Golden reference generator for the learning rules area.

Simulates all 13 synaptic learning rule classes from conex.behaviors.synapses.learning
with their corresponding dendritic input classes from conex.behaviors.synapses.dendrites,
plus the soft_bound, hard_bound, and no_bound functions.

Outputs site/tests/golden/learning.json.
"""

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pymonntorch import Network, NeuronGroup, SynapseGroup, NeuronDimension

from conex.behaviors.neurons import NeuronAxon
from conex.behaviors.network import TimeResolution, Dopamine, Payoff
from conex.behaviors.synapses import (
    SynapseInit,
    WeightInitializer,
    PreSpikeCatcher,
    PostSpikeCatcher,
    PreTrace,
    PostTrace,
    SimpleSTDP,
    SparseSTDP,
    One2OneSTDP,
    Conv2dSTDP,
    Local2dSTDP,
    SimpleiSTDP,
    One2OneiSTDP,
    SparseiSTDP,
    SimpleRSTDP,
    One2OneRSTDP,
    SparseRSTDP,
    Conv2dRSTDP,
    Local2dRSTDP,
    SimpleDendriticInput,
    SparseDendriticInput,
    One2OneDendriticInput,
    Conv2dDendriticInput,
    Local2dDendriticInput,
    soft_bound,
    hard_bound,
    no_bound,
)

OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "site", "tests", "golden", "learning.json"
)


def v(t):
    return [float(x) for x in torch.flatten(t.detach().cpu()).tolist()]


def run_class(cls, d_cls, is_sparse, is_121, is_rstdp, is_istdp, is_conv, is_local, pre_seq, post_seq):
    random.seed(42)
    torch.manual_seed(42)
    net = Network(synapse_mode="SxD")
    net.add_behavior(0, TimeResolution(dt=1.0))
    if is_rstdp:
        net.add_behavior(1, Payoff(initial_payoff=1.0))
        net.add_behavior(2, Dopamine(tau_dopamine=10.0))

    if is_conv or is_local:
        pre = NeuronGroup(
            size=NeuronDimension(depth=1, height=4, width=4),
            net=net,
            tag="pre",
            behavior=[NeuronAxon()],
        )
        post = NeuronGroup(
            size=NeuronDimension(depth=1, height=2, width=2),
            net=net,
            tag="post",
            behavior=[NeuronAxon()],
        )
    else:
        pre = NeuronGroup(size=4, net=net, tag="pre", behavior=[NeuronAxon()])
        post = NeuronGroup(size=4, net=net, tag="post", behavior=[NeuronAxon()])

    behaviors = [SynapseInit()]
    if is_sparse:
        behaviors.append(WeightInitializer(mode=0.5, density=0.5, true_sparsity=True))
    elif is_conv:
        behaviors.append(WeightInitializer(mode=0.5, weight_shape=(1, 1, 3, 3)))
    elif is_local:
        behaviors.append(
            WeightInitializer(
                mode=0.5,
                weight_shape=(1, 4, 9),
                kernel_shape=(1, 2, 2, 1, 3, 3),
            )
        )
    elif is_121:
        behaviors.append(WeightInitializer(mode=0.5, weight_shape=(4,)))
    else:
        behaviors.append(WeightInitializer(mode=0.5, weight_shape=(4, 4)))

    if is_conv or is_local:
        behaviors.append(d_cls(current_coef=1, stride=1, padding=0))
    else:
        behaviors.append(d_cls(current_coef=1))

    behaviors.extend([
        PreSpikeCatcher(),
        PostSpikeCatcher(),
        PreTrace(tau_s=10.0),
        PostTrace(tau_s=10.0),
    ])

    params = {}
    if is_istdp:
        learn = cls(lr=1e-3, rho=10.0, is_inhibitory=True)
        params = {"lr": 1e-3, "rho": 10.0, "is_inhibitory": True, "tau_s": 10.0}
    elif is_rstdp:
        learn = cls(a_plus=0.02, a_minus=0.01, tau_c=15.0, w_min=0.0, w_max=1.0)
        params = {
            "a_plus": 0.02,
            "a_minus": 0.01,
            "tau_c": 15.0,
            "w_min": 0.0,
            "w_max": 1.0,
            "tau_s": 10.0,
            "tau_dopamine": 10.0,
            "payoff": 1.0,
        }
    else:
        learn = cls(a_plus=0.02, a_minus=0.01, w_min=0.0, w_max=1.0)
        params = {
            "a_plus": 0.02,
            "a_minus": 0.01,
            "w_min": 0.0,
            "w_max": 1.0,
            "tau_s": 10.0,
        }

    behaviors.append(learn)
    syn = SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=behaviors)
    net.initialize(info=False)

    meta = {
        "is_sparse": is_sparse,
        "is_121": is_121,
        "is_rstdp": is_rstdp,
        "is_istdp": is_istdp,
        "is_conv": is_conv,
        "is_local": is_local,
    }
    if is_sparse:
        meta["src_idx"] = syn.src_idx.tolist()
        meta["dst_idx"] = syn.dst_idx.tolist()
        initial_weights = [float(x) for x in syn.weights.values().tolist()]
    elif is_conv:
        meta["kh"] = 3
        meta["kw"] = 3
        meta["stride"] = 1
        meta["padding"] = 0
        meta["srcH"] = 4
        meta["srcW"] = 4
        meta["dstH"] = 2
        meta["dstW"] = 2
        initial_weights = v(syn.weights)
    elif is_local:
        meta["kh"] = 3
        meta["kw"] = 3
        meta["stride"] = 1
        meta["padding"] = 0
        meta["srcH"] = 4
        meta["srcW"] = 4
        meta["dstH"] = 2
        meta["dstW"] = 2
        initial_weights = v(syn.weights)
    else:
        initial_weights = v(syn.weights)

    meta["initial_weights"] = initial_weights

    steps = len(pre_seq)
    history_weights = []
    history_mean_weights = []

    for t in range(steps):
        pre.spikes = torch.tensor(pre_seq[t], dtype=torch.bool)
        post.spikes = torch.tensor(post_seq[t], dtype=torch.bool)
        net.simulate_iteration()
        if is_sparse:
            cur_w = [float(x) for x in syn.weights.values().tolist()]
        else:
            cur_w = v(syn.weights)
        history_weights.append(cur_w)
        history_mean_weights.append(float(sum(cur_w) / len(cur_w)))

    return {
        "params": params,
        "meta": meta,
        "pre_spikes": pre_seq,
        "post_spikes": post_seq,
        "history_weights": history_weights,
        "history_mean_weights": history_mean_weights,
        "final_weights": history_weights[-1],
    }


def generate_bounds_data():
    w_min = 0.0
    w_max = 1.0
    w_points = [-0.2 + i * (1.4 / 14) for i in range(15)]
    w_tensor = torch.tensor(w_points, dtype=torch.float32)

    soft_vals = [float(x) for x in soft_bound(w_tensor, w_min, w_max).tolist()]
    hard_vals = [1.0 if bool(x) else 0.0 for x in hard_bound(w_tensor, w_min, w_max).tolist()]
    no_vals = [float(no_bound(x, w_min, w_max)) for x in w_points]

    # Trajectory: repeated potentiation then depression
    w0 = 0.5
    dw_plus = 0.05
    dw_minus = 0.05
    n_plus = 20
    n_minus = 20

    trajectories = {}
    for bound_name, fn in [
        ("soft_bound", soft_bound),
        ("hard_bound", lambda w, mi, ma: 1.0 if (w > mi and w < ma) else 0.0),
        ("no_bound", no_bound),
    ]:
        w = w0
        traj = [w]
        for _ in range(n_plus):
            factor = float(fn(w, w_min, w_max)) if bound_name != "soft_bound" else float(soft_bound(torch.tensor(w), w_min, w_max).item())
            w = w + dw_plus * factor
            traj.append(w)
        for _ in range(n_minus):
            factor = float(fn(w, w_min, w_max)) if bound_name != "soft_bound" else float(soft_bound(torch.tensor(w), w_min, w_max).item())
            w = w - dw_minus * factor
            traj.append(w)
        trajectories[bound_name] = traj

    return {
        "curves": {
            "w": w_points,
            "w_min": w_min,
            "w_max": w_max,
            "soft_bound": soft_vals,
            "hard_bound": hard_vals,
            "no_bound": no_vals,
        },
        "trajectory": {
            "w0": w0,
            "dw_plus": dw_plus,
            "dw_minus": dw_minus,
            "n_plus": n_plus,
            "n_minus": n_minus,
            "w_min": w_min,
            "w_max": w_max,
            "trajectories": trajectories,
        },
    }


def main():
    pre_sp_4 = [
        [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0],
        [1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0],
    ]
    post_sp_4 = [
        [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1],
        [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 1],
    ]
    pre_sp_16 = [
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    ]

    all_specs = [
        ("SimpleSTDP", SimpleSTDP, SimpleDendriticInput, False, False, False, False, False, False),
        ("SparseSTDP", SparseSTDP, SparseDendriticInput, True, False, False, False, False, False),
        ("One2OneSTDP", One2OneSTDP, One2OneDendriticInput, False, True, False, False, False, False),
        ("Conv2dSTDP", Conv2dSTDP, Conv2dDendriticInput, False, False, False, False, True, False),
        ("Local2dSTDP", Local2dSTDP, Local2dDendriticInput, False, False, False, False, False, True),
        ("SimpleiSTDP", SimpleiSTDP, SimpleDendriticInput, False, False, False, True, False, False),
        ("One2OneiSTDP", One2OneiSTDP, One2OneDendriticInput, False, True, False, True, False, False),
        ("SparseiSTDP", SparseiSTDP, SparseDendriticInput, True, False, False, True, False, False),
        ("SimpleRSTDP", SimpleRSTDP, SimpleDendriticInput, False, False, True, False, False, False),
        ("One2OneRSTDP", One2OneRSTDP, One2OneDendriticInput, False, True, True, False, False, False),
        ("SparseRSTDP", SparseRSTDP, SparseDendriticInput, True, False, True, False, False, False),
        ("Conv2dRSTDP", Conv2dRSTDP, Conv2dDendriticInput, False, False, True, False, True, False),
        ("Local2dRSTDP", Local2dRSTDP, Local2dDendriticInput, False, False, True, False, False, True),
    ]

    classes_data = {}
    for name, cls, d_cls, is_sparse, is_121, is_rstdp, is_istdp, is_conv, is_local in all_specs:
        pre_seq = pre_sp_16 if (is_conv or is_local) else pre_sp_4
        post_seq = post_sp_4
        classes_data[name] = run_class(
            cls, d_cls, is_sparse, is_121, is_rstdp, is_istdp, is_conv, is_local, pre_seq, post_seq
        )

    bounds_data = generate_bounds_data()

    result = {
        "classes": classes_data,
        "bounds": bounds_data,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"Wrote golden reference to {OUT}")


if __name__ == "__main__":
    main()
