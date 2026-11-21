"""
Golden reference numbers for the Hodgkin-Huxley plate.

Runs the REAL CoNeX HH behavior (conex.behaviors.neurons.neuron_types.hh_neurons.HH)
with float64 and writes site/tests/golden/hh.json. The JS mirror
(site/assets/js/models/hh.js) is compared against these at 1e-6 mV for v and
1e-9 for the gates.

Run from the worktree root so the local conex (with HH) is used:
    PYTHONPATH=<worktree> /Users/soroushdeimi/Desktop/myprojects/.conex-site-venv/bin/python scripts/site/golden_hh.py
"""

import json
import os

import torch

from pymonntorch import Network, NeuronGroup

from conex.behaviors.neurons.neuron_types.hh_neurons import HH
from conex.behaviors.network.time_resolution import TimeResolution

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "site", "tests", "golden", "hh.json")

DT = 0.025  # ms
PARAMS = dict(C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
              E_Na=50.0, E_K=-77.0, E_L=-54.387, threshold=0.0, init_v=-65.0)


def make_pop():
    net = Network(behavior={1: TimeResolution(dt=DT)}, dtype=torch.float64, device="cpu")
    pop = NeuronGroup(net=net, size=1, behavior={100: HH(**PARAMS)}, tag="hh")
    net.initialize()
    return net, pop


def run(current_fn, steps):
    """Run `steps` iterations; current_fn(step_index) -> uA/cm2. Record traces."""
    net, pop = make_pop()
    v, m, h, n, i_Na, i_K, spikes = [], [], [], [], [], [], []
    for i in range(steps):
        pop.I = pop.vector(mode=current_fn(i))
        v_old = pop.v.item()  # currents use the pre-step v, like CoNeX/JS
        net.simulate_iteration()
        v.append(pop.v.item())
        m.append(pop.m.item())
        h.append(pop.h.item())
        n.append(pop.n.item())
        # currents from the pre-step v and the post-step gates (CoNeX order)
        i_Na.append(PARAMS["g_Na"] * m[-1] ** 3 * h[-1] * (v_old - PARAMS["E_Na"]))
        i_K.append(PARAMS["g_K"] * n[-1] ** 4 * (v_old - PARAMS["E_K"]))
        spikes.append(bool(pop.spikes.item()))
    return {"v": v, "m": m, "h": h, "n": n, "i_Na": i_Na, "i_K": i_K, "spikes": spikes}


def main():
    golden = {"units": "mV, ms, uA/cm2, mS/cm2, uF/cm2", "dt": DT, "params": PARAMS,
              "scenarios": {}}

    # 1. Rest: I = 0 uA/cm2 for 200 ms.
    golden["scenarios"]["rest"] = {
        "current": 0.0, "steps": 8000, **run(lambda i: 0.0, 8000),
    }

    # 2. A 2 ms pulse of 10 uA/cm2 on top of rest, then 100 ms of recovery.
    pulse_steps = int(round(2.0 / DT))  # 80
    total = 2000 + pulse_steps + 4000
    golden["scenarios"]["pulse"] = {
        "pulse_amplitude": 10.0, "pulse_steps": pulse_steps, "steps": total,
        **run(lambda i: 10.0 if 2000 <= i < 2000 + pulse_steps else 0.0, total),
    }

    # 3. Constant 10 uA/cm2 for 50 ms (repetitive firing).
    golden["scenarios"]["const_10"] = {
        "current": 10.0, "steps": 2000, **run(lambda i: 10.0, 2000),
    }

    # 4. Constant 5 uA/cm2 for 200 ms (onset only, no repetitive firing).
    golden["scenarios"]["const_5"] = {
        "current": 5.0, "steps": 8000, **run(lambda i: 5.0, 8000),
    }

    with open(OUT, "w") as f:
        json.dump(golden, f, separators=(",", ":"))
        f.write("\n")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
