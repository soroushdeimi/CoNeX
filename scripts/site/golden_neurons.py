"""
Golden reference numbers for the neurons area of the site.

Builds small CoNeX networks with the real classes (LIF, ELIF, AELIF,
InherentNoise, Fire, KWTA, homeostasis, TimeResolution), runs them,
and writes site/tests/golden/neurons.json.

All quantities are in CoNeX dimensionless units (a.u.).
"""

import json
import os

import torch

from pymonntorch import Network, NeuronGroup, NeuronDimension

from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF, AELIF
from conex.behaviors.neurons.specs import Fire, KWTA, InherentNoise
from conex.behaviors.neurons.homeostasis import (
    ActivityBaseHomeostasis,
    VoltageBaseHomeostasis,
)
from conex.behaviors.network.time_resolution import TimeResolution

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "site", "tests", "golden", "neurons.json")

LIF_PARAMS = dict(R=10.0, threshold=20.0, tau=10.0, v_reset=0.0, v_rest=0.0)
ELIF_PARAMS = dict(LIF_PARAMS, delta=2.0, theta_rh=15.0)
AELIF_PARAMS = dict(ELIF_PARAMS, alpha=0.08, beta=0.5, w_tau=100.0)


def make_net(dt=1.0):
    return Network(behavior=[TimeResolution(dt=dt)])


def run_single(behavior_cls, params, current, steps, dt=1.0, extra=None):
    """Run one neuron with a constant input current; return v and spike traces."""
    net = make_net(dt)
    behavior = [behavior_cls(**params), Fire()]
    if extra is not None:
        behavior.append(extra)
    pop = NeuronGroup(net=net, size=1, behavior=behavior)
    net.initialize()
    pop.I = pop.vector(mode=current)
    v = []
    spikes = []
    omega = []
    for _ in range(steps):
        net.simulate_iteration()
        v.append(pop.v.item())
        spikes.append(bool(pop.spikes.item()))
        if hasattr(pop, "omega"):
            omega.append(pop.omega.item())
    return {"v": v, "spikes": spikes, "omega": omega}


def main():
    torch.manual_seed(0)
    golden = {"units": "a.u.", "scenarios": {}}

    # 1. LIF under a constant current.
    r = run_single(LIF, LIF_PARAMS, current=1.0, steps=100)
    golden["scenarios"]["lif_step"] = {
        "params": LIF_PARAMS, "dt": 1.0, "current": 1.0, "steps": 100,
        "v": r["v"], "spikes": r["spikes"],
    }

    # 2. Rheobase: I_rheo = (threshold - v_rest) / R = 2.0 a.u.
    #    At exactly the rheobase the LIF never fires; just above it, it does.
    r_at = run_single(LIF, LIF_PARAMS, current=2.0, steps=200)
    r_above = run_single(LIF, LIF_PARAMS, current=2.1, steps=500)
    golden["scenarios"]["lif_rheobase"] = {
        "params": LIF_PARAMS, "dt": 1.0,
        "i_rheo": (LIF_PARAMS["threshold"] - LIF_PARAMS["v_rest"]) / LIF_PARAMS["R"],
        "at_rheobase": {"current": 2.0, "steps": 200,
                        "spike_count": sum(r_at["spikes"]),
                        "v_final": r_at["v"][-1]},
        "above_rheobase": {"current": 2.1, "steps": 500,
                           "spike_count": sum(r_above["spikes"]),
                           "first_spike_step": r_above["spikes"].index(True)},
    }

    # 3. ELIF under a constant current.
    r = run_single(ELIF, ELIF_PARAMS, current=1.0, steps=100)
    golden["scenarios"]["elif_step"] = {
        "params": ELIF_PARAMS, "dt": 1.0, "current": 1.0, "steps": 100,
        "v": r["v"], "spikes": r["spikes"],
    }

    # 4. AELIF under a constant current (adaptation trace).
    r = run_single(AELIF, AELIF_PARAMS, current=1.0, steps=200)
    golden["scenarios"]["aelif_step"] = {
        "params": AELIF_PARAMS, "dt": 1.0, "current": 1.0, "steps": 200,
        "v": r["v"], "spikes": r["spikes"], "omega": r["omega"],
    }

    # 5. f-I curves: spike counts for each model at several currents.
    currents = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    fi = {}
    for name, cls, params in [
        ("lif", LIF, LIF_PARAMS),
        ("elif", ELIF, ELIF_PARAMS),
        ("aelif", AELIF, AELIF_PARAMS),
    ]:
        counts = []
        for c in currents:
            r = run_single(cls, params, current=c, steps=2000)
            counts.append(sum(r["spikes"]))
        fi[name] = counts
    golden["scenarios"]["fi_curves"] = {
        "currents": currents, "steps": 2000, "dt": 1.0, "counts": fi,
    }

    # 6. Activity-based homeostasis on a group of 4 neurons.
    #    Current 3.0 a.u. is above rheobase so the group actually fires and
    #    the threshold is driven down toward the target activity.
    torch.manual_seed(0)
    net = make_net(1.0)
    pop = NeuronGroup(net=net, size=4, behavior=[
        LIF(**LIF_PARAMS),
        Fire(),
        ActivityBaseHomeostasis(activity_rate=2, window_size=10,
                                updating_rate=0.1, decay_rate=1.0),
    ])
    net.initialize()
    pop.threshold = pop.vector(mode=20.0)
    pop.I = pop.vector(mode=3.0)
    window_size = 10
    threshold_trace = []
    window_spikes = []
    acc = 0
    for i in range(40):
        net.simulate_iteration()
        acc += int(pop.spikes.sum().item())
        if (i + 1) % window_size == 0:
            threshold_trace.append(pop.threshold.tolist())
            window_spikes.append(acc)
            acc = 0
    golden["scenarios"]["homeostasis_activity"] = {
        "params": LIF_PARAMS,
        "homeostasis": {"activity_rate": 2, "window_size": 10,
                        "updating_rate": 0.1, "decay_rate": 1.0},
        "size": 4, "current": 3.0, "steps": 40, "dt": 1.0,
        "threshold_trace": threshold_trace,
        "window_spikes": window_spikes,
    }

    # 7. Voltage-based homeostasis on a single neuron.
    torch.manual_seed(0)
    net = make_net(1.0)
    pop = NeuronGroup(net=net, size=1, behavior=[
        LIF(**LIF_PARAMS),
        Fire(),
        VoltageBaseHomeostasis(target_voltage=5.0, eta_ip=0.001),
    ])
    net.initialize()
    pop.I = pop.vector(mode=1.0)
    v = []
    exhaustion = []
    for _ in range(100):
        net.simulate_iteration()
        v.append(pop.v.item())
        exhaustion.append(pop.exhaustion.item())
    golden["scenarios"]["homeostasis_voltage"] = {
        "params": LIF_PARAMS,
        "homeostasis": {"target_voltage": 5.0, "eta_ip": 0.001},
        "current": 1.0, "steps": 100, "dt": 1.0,
        "v": v, "exhaustion": exhaustion,
    }

    # 8. KWTA as a pure function: given a fixed voltage vector, the k winners
    #    (top-k by v) keep their voltage and every other spiking neuron is reset
    #    to v_reset. If at most k neurons are above threshold, nothing changes.
    #    This is deterministic, so the JS mirror can be compared exactly.
    net = make_net(1.0)
    kwta = KWTA(k=2)
    pop = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=1, width=8),
                      behavior=[LIF(**LIF_PARAMS), kwta])
    net.initialize()

    def kwta_case(v_in):
        pop.v = torch.tensor(v_in)
        kwta.forward(pop)
        return pop.v.tolist()

    golden["scenarios"]["kwta"] = {
        "params": LIF_PARAMS,
        "kwta": {"k": 2},
        "size": 8,
        "cases": [
            {"v_in": [25.0, 22.0, 21.0, 20.5, 19.0, 18.0, 17.0, 16.0],
             "v_out": kwta_case([25.0, 22.0, 21.0, 20.5, 19.0, 18.0, 17.0, 16.0])},
            {"v_in": [25.0, 22.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0],
             "v_out": kwta_case([25.0, 22.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0])},
            {"v_in": [25.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0],
             "v_out": kwta_case([25.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0])},
        ],
    }

    # 9. KWTA under InherentNoise (torch RNG) — a reference trace for the plate
    #    only; the JS plate uses its own seeded PRNG and is not compared to this.
    torch.manual_seed(0)
    net = make_net(1.0)
    pop = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=1, width=8),
                      behavior=[
                          LIF(**LIF_PARAMS),
                          InherentNoise(mode="rand", scale=1.0, offset=0.0),
                          Fire(),
                          KWTA(k=2),
                      ])
    net.initialize()
    pop.I = pop.vector(mode=3.0)
    v_trace = []
    spike_trace = []
    for _ in range(10):
        net.simulate_iteration()
        v_trace.append(pop.v.tolist())
        spike_trace.append(pop.spikes.tolist())
    golden["scenarios"]["kwta_noisy_reference"] = {
        "params": LIF_PARAMS,
        "kwta": {"k": 2},
        "noise": {"mode": "rand", "scale": 1.0, "offset": 0.0},
        "size": 8, "current": 3.0, "steps": 10, "dt": 1.0,
        "v": v_trace, "spikes": spike_trace,
    }

    with open(OUT, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
