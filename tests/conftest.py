"""Shared helpers for building minimal networks in tests."""

import pytest
import torch
from pymonntorch import Network, NeuronGroup

from conex import TimeResolution


@pytest.fixture
def make_group():
    """Build a NeuronGroup on a network that carries a time resolution.

    Behaviors across CoNeX read ``network.dt``, which only exists once
    TimeResolution has run, so every test network needs it.
    """

    def build(size, behavior, dt=1.0, tag="test"):
        net = Network(behavior={1: TimeResolution(dt=dt)}, dtype=torch.float32, device="cpu")
        neurons = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
        net.initialize()
        return neurons

    return build
