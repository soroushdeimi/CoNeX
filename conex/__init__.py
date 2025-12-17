"""
CoNeX: Cortical Network for Everything.

A framework for building spiking neural networks with cortical structures,
including support for Thousand Brains Theory components.
"""

from conex.behaviors import *
from conex.nn import *
from conex.helpers import *

__version__ = "0.1.5"

__all__ = [
    "__version__",
    # Structures
    "Container",
    "Layer",
    "CorticalLayer",
    "CorticalColumn",
    "Neocortex",
    "InputLayer",
    "OutputLayer",
    "Synapsis",
    "Port",
    # Utilities
    "prioritize_behaviors",
]
