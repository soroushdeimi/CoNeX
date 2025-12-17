"""Synapse behaviors for CoNeX."""

from .specs import *
from .learning import *
from .dendrites import *
from .predictive import (
    PredictiveSynapseInit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    LateralPredictionSynapse,
    PredictiveSynapseLearning,
    PredictiveConnection,
)

__all__ = [
    # Specs
    "SynapseInit",
    "DelayInitializer",
    "WeightInitializer",
    "WeightNormalization",
    "CurrentNormalization",
    "WeightClip",
    "PreSpikeCatcher",
    "PostSpikeCatcher",
    "PreTrace",
    "PostTrace",
    # Learning
    "BaseLearning",
    "SimpleSTDP",
    "SparseSTDP",
    "One2OneSTDP",
    "SimpleiSTDP",
    "One2OneiSTDP",
    "SparseiSTDP",
    "Conv2dSTDP",
    "Local2dSTDP",
    "SimpleRSTDP",
    "One2OneRSTDP",
    # Dendrites
    "BaseDendriticInput",
    "SparseDendriticInput",
    "One2OneDendriticInput",
    "SimpleDendriticInput",
    "AveragePool2D",
    "LateralDendriticInput",
    "Conv2dDendriticInput",
    "Local2dDendriticInput",
    # Predictive Coding
    "PredictiveSynapseInit",
    "FeedbackPredictionSynapse",
    "FeedforwardErrorSynapse",
    "LateralPredictionSynapse",
    "PredictiveSynapseLearning",
    "PredictiveConnection",
]
