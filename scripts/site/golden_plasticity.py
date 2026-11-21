"""Golden numbers for the plasticity area of the site.

Builds small CoNeX networks with the real classes, runs them with scripted
spike trains, and writes site/tests/golden/plasticity.json.
"""

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pymonntorch import NeuronGroup, SynapseGroup, NeuronDimension, Network

from conex.behaviors.neurons import NeuronAxon
from conex.behaviors.synapses import (
    SynapseInit,
    WeightInitializer,
    PreSpikeCatcher,
    PostSpikeCatcher,
    PreTrace,
    PostTrace,
    SimpleSTDP,
    SimpleiSTDP,
    SimpleRSTDP,
    SimpleDendriticInput,
    SparseDendriticInput,
    One2OneDendriticInput,
    Conv2dDendriticInput,
    Local2dDendriticInput,
    LateralDendriticInput,
    AveragePool2D,
)
from conex.behaviors.network import Dopamine, Payoff, TimeResolution
from conex.helpers.transforms.encoders import (
    SimplePoisson,
    Poisson,
    Intensity2Latency,
)

OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "site", "tests", "golden", "plasticity.json"
)


def v(t):
    # flatten any shape to a row-major list of floats
    return [float(x) for x in torch.flatten(t.detach().cpu()).tolist()]


def make_net():
    return Network(synapse_mode="SxD")


def lif_group(net, size, tag):
    # Scripted-spike group: only NeuronAxon is needed so axon.get_spike works.
    # A real LIF with threshold=0, v_rest=0 fires every step and would clobber
    # the scripted spikes, so it is intentionally omitted here.
    return NeuronGroup(
        size=size,
        net=net,
        tag=tag,
        behavior=[NeuronAxon()],
    )


def stdp_synapse(net, src, dst, a_plus, a_minus, w0, tau_s, **kw):
    return SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=[
            SynapseInit(),
            WeightInitializer(mode=w0),
            SimpleDendriticInput(current_coef=1),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=tau_s),
            PostTrace(tau_s=tau_s),
            SimpleSTDP(a_plus=a_plus, a_minus=a_minus, **kw),
        ],
    )


def drive(net, src, dst, pre, post, steps):
    """Run `steps` iterations with scripted spikes; return recorded lists."""
    rec = {"w": [], "pre_trace": [], "post_trace": [], "post_spikes": []}
    syn = net.SynapseGroups[0]
    for t in range(steps):
        src.spikes = torch.tensor(pre[t], dtype=torch.bool)
        dst.spikes = torch.tensor(post[t], dtype=torch.bool)
        net.simulate_iteration()
        rec["w"].append(v(syn.weights))
        rec["pre_trace"].append(v(syn.pre_trace))
        rec["post_trace"].append(v(syn.post_trace))
        rec["post_spikes"].append(v(dst.spikes))
    return rec


def scenario_stdp():
    """SimpleSTDP on a 2x2 synapse with scripted pre/post spike trains."""
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, 2, "pre")
    post = lif_group(net, 2, "post")
    stdp_synapse(
        net, pre, post, a_plus=0.01, a_minus=0.002, w0=0.5, tau_s=10.0
    )
    net.initialize(info=False)
    # pre spikes: neuron 0 at t=0,1,2; neuron 1 at t=5,6
    # post spikes: neuron 0 at t=3,4; neuron 1 at t=8
    pre_sp = [[0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1]]
    post_sp = [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]
    rec = drive(net, pre, post, pre_sp, post_sp, 10)
    return {
        "params": {
            "a_plus": 0.01, "a_minus": 0.002, "w_min": 0.0, "w_max": 1.0,
            "tau_s": 10.0, "dt": 1.0, "w0": 0.5,
            "positive_bound": "no_bound", "negative_bound": "no_bound",
        },
        "pre_spikes": pre_sp,
        "post_spikes_in": post_sp,
        "weights": rec["w"],
        "pre_trace": rec["pre_trace"],
        "post_trace": rec["post_trace"],
        "post_spikes_out": rec["post_spikes"],
    }


def istdp_synapse(net, src, dst, rho, lr, w0, tau_s):
    return SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal,inh",
        behavior=[
            SynapseInit(),
            WeightInitializer(mode=w0),
            SimpleDendriticInput(current_coef=1),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=tau_s),
            PostTrace(tau_s=tau_s),
            SimpleiSTDP(rho=rho, lr=lr),
        ],
    )


def scenario_istdp():
    """SimpleiSTDP on a 2x2 inhibitory synapse with scripted spike trains."""
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, 2, "pre")
    post = lif_group(net, 2, "post")
    istdp_synapse(net, pre, post, rho=0.05, lr=1e-5, w0=0.5, tau_s=10.0)
    net.initialize(info=False)
    # pre spikes: neuron 0 at t=0,1,2,3; neuron 1 at t=6
    # post spikes: neuron 0 at t=4,5; neuron 1 at t=8
    pre_sp = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]]
    post_sp = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0]]
    rec = drive(net, pre, post, pre_sp, post_sp, 10)
    syn = net.SynapseGroups[0]
    return {
        "params": {
            "rho": 0.05, "lr": 1e-5, "is_inhibitory": True,
            "tau_s": 10.0, "dt": 1.0, "w0": 0.5,
        },
        "alpha": float(syn.SimpleiSTDP.alpha),
        "pre_spikes": pre_sp,
        "post_spikes_in": post_sp,
        "weights": rec["w"],
        "pre_trace": rec["pre_trace"],
        "post_trace": rec["post_trace"],
    }


def rstdp_synapse(net, src, dst, a_plus, a_minus, tau_c, w0, tau_s):
    return SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=[
            SynapseInit(),
            WeightInitializer(mode=w0),
            SimpleDendriticInput(current_coef=1),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=tau_s),
            PostTrace(tau_s=tau_s),
            SimpleRSTDP(a_plus=a_plus, a_minus=a_minus, tau_c=tau_c),
        ],
    )


def scenario_rstdp(reward):
    """SimpleRSTDP with Dopamine + Payoff; reward is a per-step payoff list."""
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    net.add_behavior(1, Payoff(initial_payoff=0.0))
    net.add_behavior(2, Dopamine(tau_dopamine=10.0))
    pre = lif_group(net, 2, "pre")
    post = lif_group(net, 2, "post")
    rstdp_synapse(net, pre, post, a_plus=0.01, a_minus=0.002, tau_c=10.0, w0=0.5, tau_s=10.0)
    net.initialize(info=False)
    steps = 10
    pre_sp = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]]
    post_sp = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0]]
    rec = {"w": [], "c": [], "dopamine": [], "pre_trace": [], "post_trace": []}
    syn = net.SynapseGroups[0]
    for t in range(steps):
        net.payoff = reward[t]
        pre.spikes = torch.tensor(pre_sp[t], dtype=torch.bool)
        post.spikes = torch.tensor(post_sp[t], dtype=torch.bool)
        net.simulate_iteration()
        rec["w"].append(v(syn.weights))
        rec["c"].append(v(syn.c))
        rec["dopamine"].append(float(net.dopamine_concentration))
        rec["pre_trace"].append(v(syn.pre_trace))
        rec["post_trace"].append(v(syn.post_trace))
    return {
        "params": {
            "a_plus": 0.01, "a_minus": 0.002, "tau_c": 10.0, "tau_s": 10.0,
            "dt": 1.0, "w0": 0.5, "tau_dopamine": 10.0,
        },
        "payoff": reward,
        "pre_spikes": pre_sp,
        "post_spikes_in": post_sp,
        "weights": rec["w"],
        "eligibility": rec["c"],
        "dopamine": rec["dopamine"],
        "pre_trace": rec["pre_trace"],
        "post_trace": rec["post_trace"],
    }


def pattern_16x16():
    # digit-like stroke: a "7" drawn on a 16x16 grid, values in [0, 1]
    img = torch.zeros(16, 16)
    for x in range(16):
        img[2, x] = 1.0
        img[3, x] = 0.6
    for y in range(2, 14):
        x = int(13 - (y - 2) * 0.8)
        img[y, x] = 1.0
        if x + 1 < 16:
            img[y, x + 1] = 0.6
    return img


def scenario_encoders():
    torch.manual_seed(42)
    img = pattern_16x16()
    tw, ratio = 30, 5 / 30
    simple = SimplePoisson(time_window=tw, ratio=ratio)(img)
    torch.manual_seed(42)
    poisson = Poisson(time_window=tw, ratio=ratio)(img)
    i2l = Intensity2Latency(time_window=tw)(img)
    return {
        "params": {"time_window": tw, "ratio": ratio, "seed": 42},
        "image": v(img),
        "simple_poisson": v(simple),
        "poisson": v(poisson),
        "intensity2latency": v(i2l),
    }


def run_input(net, pre, spikes, steps=2):
    """Set scripted pre spikes, run `steps` iterations, return (I, weights).

    DendriticInput (key 2) runs before PreSpikeCatcher (key 3), so I lags the
    spikes by one step; run 2 iterations with the same pattern to settle.
    """
    syn = net.SynapseGroups[0]
    for ng in net.NeuronGroups:
        ng.spikes = torch.zeros(ng.size, dtype=torch.bool)
    for t in range(steps):
        pre.spikes = torch.tensor(spikes[0], dtype=torch.bool)
        net.simulate_iteration()
    if not hasattr(syn, "weights"):
        return v(syn.I), None
    try:
        w = syn.weights.to_dense()
    except AttributeError:
        w = syn.weights
    return v(syn.I), v(w)


def scenario_connections():
    out = {}
    # dense: 4x4 -> 3
    torch.manual_seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, 3, "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(), WeightInitializer(mode="random"),
        SimpleDendriticInput(current_coef=1), PreSpikeCatcher()])
    net.initialize(info=False)
    spikes = [[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
    i, w = run_input(net, pre, spikes)
    out["dense"] = {"weights": w, "current": i, "spikes": spikes[0]}

    # sparse: 4x4 -> 3, density 0.5 (WeightInitializer uses random.sample)
    torch.manual_seed(7)
    random.seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, 3, "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(), WeightInitializer(mode="random", density=0.5),
        SparseDendriticInput(current_coef=1), PreSpikeCatcher()])
    net.initialize(info=False)
    i, w = run_input(net, pre, spikes)
    out["sparse"] = {"weights": w, "current": i, "spikes": spikes[0]}

    # one2one: 4x4 -> 4x4
    torch.manual_seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(), WeightInitializer(mode="random", weight_shape=(16,)),
        One2OneDendriticInput(current_coef=1), PreSpikeCatcher()])
    net.initialize(info=False)
    i, w = run_input(net, pre, spikes)
    out["one2one"] = {"weights": w, "current": i, "spikes": spikes[0]}
    return out


def scenario_connections_2d():
    out = {}
    spikes = [[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
    # conv2d: 4x4x1 -> 2x2x1, kernel 3x3, stride 1, padding 0
    torch.manual_seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, NeuronDimension(depth=1, height=2, width=2), "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(),
        WeightInitializer(mode="random", weight_shape=(1, 1, 3, 3)),
        Conv2dDendriticInput(current_coef=1, stride=1, padding=0),
        PreSpikeCatcher()])
    net.initialize(info=False)
    i, w = run_input(net, pre, spikes)
    out["conv2d"] = {"weights": w, "current": i, "spikes": spikes[0]}

    # local2d: 4x4x1 -> 2x2x1, kernel 3x3
    torch.manual_seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, NeuronDimension(depth=1, height=2, width=2), "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(),
        WeightInitializer(mode="random", weight_shape=(1, 4, 9),
                          kernel_shape=(1, 2, 2, 1, 3, 3)),
        Local2dDendriticInput(current_coef=1, stride=1, padding=0),
        PreSpikeCatcher()])
    net.initialize(info=False)
    i, w = run_input(net, pre, spikes)
    out["local2d"] = {"weights": w, "current": i, "spikes": spikes[0]}

    # lateral: 4x4x1 -> itself, kernel 3x3
    torch.manual_seed(7)
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    SynapseGroup(net=net, src=pre, dst=pre, tag="Proximal", behavior=[
        SynapseInit(),
        WeightInitializer(mode="random", weight_shape=(1, 1, 1, 3, 3)),
        LateralDendriticInput(current_coef=1),
        PreSpikeCatcher()])
    net.initialize(info=False)
    i, w = run_input(net, pre, spikes)
    out["lateral"] = {"weights": w, "current": i, "spikes": spikes[0]}

    # average pool: 4x4x1 -> 2x2x1
    net = make_net()
    net.add_behavior(0, TimeResolution(dt=1.0))
    pre = lif_group(net, NeuronDimension(depth=1, height=4, width=4), "pre")
    post = lif_group(net, NeuronDimension(depth=1, height=2, width=2), "post")
    SynapseGroup(net=net, src=pre, dst=post, tag="Proximal", behavior=[
        SynapseInit(),
        AveragePool2D(current_coef=1),
        PreSpikeCatcher()])
    net.initialize(info=False)
    i, _ = run_input(net, pre, spikes)
    out["avgpool"] = {"current": i, "spikes": spikes[0]}
    return out


def main():
    data = {
        "stdp": scenario_stdp(),
        "istdp": scenario_istdp(),
        "rstdp_reward": scenario_rstdp([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "rstdp_noreward": scenario_rstdp([0.0] * 10),
        "encoders": scenario_encoders(),
        "connections": {**scenario_connections(), **scenario_connections_2d()},
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
