"""Tests for the Hodgkin-Huxley (1952) neuron behavior.

All expected numbers were measured by running the library with the default
parameters (C=1 uF/cm2, g_Na=120, g_K=36, g_L=0.3 mS/cm2, E_Na=50, E_K=-77,
E_L=-54.387 mV, threshold=0 mV, init_v=-65 mV) at dt=0.025 ms.
"""

# Measured firing behavior (dt=0.025 ms, init_v=-65 mV):
#   10 uA/cm2, 200 ms -> 14 spikes (70 Hz), periodic (last-3 ISI spread 0.17%)
#   6  uA/cm2, 200 ms -> 2 spikes at 2.575 ms and 21.9 ms (onset transient)
#   5  uA/cm2, 200 ms -> 1 spike (onset only, no repetitive firing)

import numpy as np
import pytest
import torch

from conex import HH

DT = 0.025  # ms


def hh_rates(v):
    """Hand-computed HH (1952) gate rates in ms^-1 at voltage v (mV)."""
    u = (v + 40.0) / 10.0
    alpha_m = -u / np.expm1(-u)
    beta_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    q = (v + 55.0) / 10.0
    alpha_n = 0.1 * (-q / np.expm1(-q))
    beta_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def run(net, pop, steps):
    """Simulate and accumulate spike counts and the peak voltage."""
    n_spikes = 0
    peak = -np.inf
    for _ in range(steps):
        net.simulate_iteration()
        n_spikes += int(pop.spikes.sum())
        peak = max(peak, pop.v.item())
    return n_spikes, peak


def spike_times(net, pop, steps):
    """Simulate and record the time (ms) of each spike."""
    times = []
    for t in range(steps):
        net.simulate_iteration()
        if pop.spikes[0]:
            times.append((t + 1) * DT)
    return times


@pytest.fixture
def hh_group(make_group):
    return make_group(1, HH(init_v=-65.0), dt=DT, tag="hh")


class TestHH:
    def test_resting_potential_settles_near_minus_65(self, hh_group):
        # measured: -64.9966 mV after 200 ms with I = 0 uA/cm2
        hh_group.I = hh_group.vector(mode=0.0)
        run(hh_group.network, hh_group, 8000)
        assert hh_group.v.item() == pytest.approx(-65.0, abs=0.1)

    def test_gate_steady_states_at_resting_potential(self, hh_group):
        # hand-computed alpha/(alpha+beta) at -65 mV
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = hh_rates(-65.0)
        assert hh_group.m.item() == pytest.approx(
            alpha_m / (alpha_m + beta_m), abs=1e-5
        )
        assert hh_group.h.item() == pytest.approx(
            alpha_h / (alpha_h + beta_h), abs=1e-5
        )
        assert hh_group.n.item() == pytest.approx(
            alpha_n / (alpha_n + beta_n), abs=1e-5
        )

    def test_single_suprathreshold_pulse_gives_one_spike(self, hh_group):
        # 10 uA/cm2 for 2 ms: one spike, peak 39.33 mV (measured)
        hh_group.I = hh_group.vector(mode=0.0)
        run(hh_group.network, hh_group, 2000)  # settle
        hh_group.I = hh_group.vector(mode=10.0)
        n_spikes, peak = run(hh_group.network, hh_group, 80)  # 2 ms
        hh_group.I = hh_group.vector(mode=0.0)
        n_spikes += run(hh_group.network, hh_group, 4000)[0]
        assert n_spikes == 1
        assert peak > 30.0  # mV

    def test_constant_current_fires_periodically(self, hh_group):
        # 10 uA/cm2 for 200 ms: 14 spikes (70 Hz, measured). The last three
        # inter-spike intervals agree within 1%, i.e. the firing is periodic,
        # not an onset transient.
        hh_group.I = hh_group.vector(mode=10.0)
        times = spike_times(hh_group.network, hh_group, 8000)
        assert len(times) == 14
        isis = [b - a for a, b in zip(times, times[1:])]
        last3 = isis[-3:]
        spread = (max(last3) - min(last3)) / (sum(last3) / 3)
        assert spread < 0.01

    def test_current_near_rheobase_gives_an_onset_transient(self, hh_group):
        # 6 uA/cm2 for 200 ms: 2 spikes (measured), both early (2.575 ms and
        # 21.9 ms) -- an onset transient, not sustained periodic firing.
        hh_group.I = hh_group.vector(mode=6.0)
        times = spike_times(hh_group.network, hh_group, 8000)
        assert len(times) == 2
        assert times[-1] < 50.0  # both spikes in the first quarter

    def test_current_below_rheobase_does_not_fire_repeatedly(self, hh_group):
        # 5 uA/cm2 for 200 ms: at most one onset spike (measured: 1)
        hh_group.I = hh_group.vector(mode=5.0)
        times = spike_times(hh_group.network, hh_group, 8000)
        assert len(times) <= 1

    def test_no_reset_after_spike(self, hh_group):
        # HH has no reset: after the spike the voltage decays back to rest
        hh_group.I = hh_group.vector(mode=0.0)
        run(hh_group.network, hh_group, 2000)
        hh_group.I = hh_group.vector(mode=10.0)
        run(hh_group.network, hh_group, 80)
        hh_group.I = hh_group.vector(mode=0.0)
        run(hh_group.network, hh_group, 4000)
        assert hh_group.v.item() == pytest.approx(-65.0, abs=0.1)

    def test_gate_rates_are_finite_at_singularity_voltages(self, make_group):
        # alpha_m is singular at v = -40 mV and alpha_n at v = -55 mV
        for v0 in (-40.0, -55.0):
            pop = make_group(1, HH(init_v=v0), dt=DT, tag="hh")
            pop.I = pop.vector(mode=0.0)
            run(pop.network, pop, 100)
            assert torch.isfinite(pop.v).all()
            assert torch.isfinite(pop.m).all()
            assert torch.isfinite(pop.h).all()
            assert torch.isfinite(pop.n).all()

    def test_matches_standalone_numpy_euler(self):
        # same equations, same Euler scheme, float64 on both sides over 50 ms
        from pymonntorch import Network, NeuronGroup

        from conex import TimeResolution

        net = Network(
            behavior={1: TimeResolution(dt=DT)},
            dtype=torch.float64,
            device="cpu",
        )
        pop = NeuronGroup(net=net, size=1, behavior={100: HH(init_v=-65.0)}, tag="hh")
        net.initialize()
        pop.I = pop.vector(mode=0.0)

        v = np.float64(-65.0)
        m = np.float64(pop.m.item())
        h = np.float64(pop.h.item())
        n = np.float64(pop.n.item())
        C, g_Na, g_K, g_L = 1.0, 120.0, 36.0, 0.3
        E_Na, E_K, E_L = 50.0, -77.0, -54.387
        I = 0.0

        for _ in range(2000):  # 50 ms
            a_m, b_m, a_h, b_h, a_n, b_n = hh_rates(v)
            m += (a_m * (1.0 - m) - b_m * m) * DT
            h += (a_h * (1.0 - h) - b_h * h) * DT
            n += (a_n * (1.0 - n) - b_n * n) * DT
            i_Na = g_Na * m**3 * h * (v - E_Na)
            i_K = g_K * n**4 * (v - E_K)
            i_L = g_L * (v - E_L)
            v += (I - i_Na - i_K - i_L) * DT / C

            pop.network.simulate_iteration()

        assert abs(pop.v.item() - v) < 1e-9
        assert abs(pop.m.item() - m) < 1e-9
        assert abs(pop.h.item() - h) < 1e-9
        assert abs(pop.n.item() - n) < 1e-9
