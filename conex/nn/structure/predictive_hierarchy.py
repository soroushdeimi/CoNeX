"""Predictive Coding Hierarchy for CoNeX.

Provides high-level structures for building complete predictive
coding networks with multiple hierarchical levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import torch
from torch import Tensor
from pymonntorch import Network, Behavior

from ..structure.layer import CorticalLayer
from ..structure.synapsis import Synapsis
from ...behaviors.neurons.predictive_coding import (
    PredictiveCodingConfig,
    PredictionUnit,
    ErrorUnit,
    PrecisionWeighting,
    PredictiveCodingLearning,
    FreeEnergyMinimization,
    create_predictive_layer,
)
from ...behaviors.synapses.predictive import (
    PredictiveSynapseInit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    PredictiveSynapseLearning,
)


@dataclass
class HierarchyLevel:
    """Represents one level in the predictive coding hierarchy.
    
    Attributes:
        size: Number of neurons at this level.
        name: Name identifier for this level.
        layer: The actual CorticalLayer (set during construction).
        config: Predictive coding config for this level.
    """
    
    size: int
    name: str = ""
    layer: Optional[CorticalLayer] = None
    config: Optional[PredictiveCodingConfig] = None


class PredictiveHierarchy:
    """Multi-level predictive coding hierarchy.
    
    Creates a complete predictive coding network with:
    - Multiple hierarchical levels (sensory → abstract)
    - Bidirectional connections (feedforward errors, feedback predictions)
    - Precision weighting at each level
    - Free energy tracking
    
    Based on:
    - Rao & Ballard (1999): Predictive coding in visual cortex
    - Bastos et al. (2012): Canonical microcircuits for predictive coding
    
    Example:
        >>> net = Network()
        >>> hierarchy = PredictiveHierarchy(
        ...     net,
        ...     level_sizes=[784, 256, 64, 16],
        ...     level_names=["sensory", "L4", "L2/3", "abstract"]
        ... )
        >>> hierarchy.build()
        >>> # Set sensory input
        >>> hierarchy.set_input(sensory_data)
        >>> net.simulate(100)
        >>> # Get free energy
        >>> print(hierarchy.get_total_free_energy())
    
    Args:
        network: PyMonntorch Network.
        level_sizes: List of sizes for each level (bottom to top).
        level_names: Optional names for each level.
        config: Base PredictiveCodingConfig.
        build_immediately: Whether to build hierarchy on construction.
    """
    
    def __init__(
        self,
        network: Network,
        level_sizes: List[int],
        level_names: Optional[List[str]] = None,
        config: Optional[PredictiveCodingConfig] = None,
        build_immediately: bool = True,
    ):
        self.network = network
        self.config = config or PredictiveCodingConfig()
        self.n_levels = len(level_sizes)
        
        # Create level specs
        if level_names is None:
            level_names = [f"level_{i}" for i in range(self.n_levels)]
        
        self.levels: List[HierarchyLevel] = [
            HierarchyLevel(size=size, name=name, config=self.config)
            for size, name in zip(level_sizes, level_names)
        ]
        
        # Connection storage
        self.feedforward_synapses: List[Synapsis] = []
        self.feedback_synapses: List[Synapsis] = []
        
        # Build if requested
        if build_immediately:
            self.build()
    
    def build(self) -> "PredictiveHierarchy":
        """Build the complete predictive hierarchy.
        
        Returns:
            Self for chaining.
        """
        # Create layers
        for level in self.levels:
            behaviors = self._create_level_behaviors(level)
            level.layer = CorticalLayer(
                net=self.network,
                size=level.size,
                behavior=behaviors,
                tag=level.name,
            )
        
        # Create connections
        for i in range(self.n_levels - 1):
            lower = self.levels[i]
            higher = self.levels[i + 1]
            
            # Feedforward (error) connection: lower → higher
            ff_syn = self._create_feedforward_synapse(lower.layer, higher.layer)
            self.feedforward_synapses.append(ff_syn)
            
            # Feedback (prediction) connection: higher → lower
            fb_syn = self._create_feedback_synapse(higher.layer, lower.layer)
            self.feedback_synapses.append(fb_syn)
        
        return self
    
    def _create_level_behaviors(self, level: HierarchyLevel) -> Dict[int, Behavior]:
        """Create behaviors for a hierarchy level."""
        cfg = level.config or self.config
        
        return {
            100: PredictionUnit(config=cfg),
            110: ErrorUnit(config=cfg, precision_weighted=True),
            120: PrecisionWeighting(config=cfg, learnable=True),
            130: PredictiveCodingLearning(config=cfg),
            140: FreeEnergyMinimization(config=cfg, track_components=True),
        }
    
    def _create_feedforward_synapse(
        self,
        src_layer: CorticalLayer,
        dst_layer: CorticalLayer,
    ) -> Synapsis:
        """Create feedforward (error propagation) synapse."""
        behaviors = {
            100: PredictiveSynapseInit(
                config=self.config, connection_type="feedforward"
            ),
            110: FeedforwardErrorSynapse(
                config=self.config, precision_weighted=True
            ),
            120: PredictiveSynapseLearning(
                config=self.config, learning_type="error_driven"
            ),
        }
        
        return Synapsis(
            net=self.network,
            src=src_layer,
            dst=dst_layer,
            behavior=behaviors,
            tag=f"ff_{src_layer.tag}_{dst_layer.tag}",
        )
    
    def _create_feedback_synapse(
        self,
        src_layer: CorticalLayer,
        dst_layer: CorticalLayer,
    ) -> Synapsis:
        """Create feedback (prediction) synapse."""
        behaviors = {
            100: PredictiveSynapseInit(
                config=self.config, connection_type="feedback"
            ),
            110: FeedbackPredictionSynapse(config=self.config),
            120: PredictiveSynapseLearning(
                config=self.config, learning_type="error_driven"
            ),
        }
        
        return Synapsis(
            net=self.network,
            src=src_layer,
            dst=dst_layer,
            behavior=behaviors,
            tag=f"fb_{src_layer.tag}_{dst_layer.tag}",
        )
    
    def set_input(self, data: Tensor) -> None:
        """Set sensory input to the bottom level.
        
        Args:
            data: Input tensor matching bottom level size.
        """
        bottom_layer = self.levels[0].layer
        if bottom_layer is not None:
            bottom_layer.activity = data.to(
                device=self.network.device,
                dtype=self.network.def_dtype,
            )
    
    def get_predictions(self, level: int = 0) -> Tensor:
        """Get predictions at a specific level.
        
        Args:
            level: Level index (0 = bottom/sensory).
        
        Returns:
            Prediction tensor.
        """
        layer = self.levels[level].layer
        if layer is not None and hasattr(layer, "prediction"):
            return layer.prediction
        return torch.zeros(self.levels[level].size)
    
    def get_errors(self, level: int = 0) -> Tensor:
        """Get prediction errors at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Error tensor.
        """
        layer = self.levels[level].layer
        if layer is not None and hasattr(layer, "prediction_error"):
            return layer.prediction_error
        return torch.zeros(self.levels[level].size)
    
    def get_free_energy(self, level: int = 0) -> float:
        """Get free energy at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Free energy value.
        """
        layer = self.levels[level].layer
        if layer is not None and hasattr(layer, "free_energy"):
            return layer.free_energy.item()
        return 0.0
    
    def get_total_free_energy(self) -> float:
        """Get total free energy across all levels.
        
        Returns:
            Sum of free energy at all levels.
        """
        return sum(self.get_free_energy(i) for i in range(self.n_levels))
    
    def get_precision(self, level: int = 0) -> Tensor:
        """Get precision estimates at a specific level.
        
        Args:
            level: Level index.
        
        Returns:
            Precision tensor.
        """
        layer = self.levels[level].layer
        if layer is not None and hasattr(layer, "precision"):
            return layer.precision
        return torch.ones(self.levels[level].size)
    
    def __repr__(self) -> str:
        """String representation of hierarchy."""
        level_strs = [f"{l.name}({l.size})" for l in self.levels]
        return f"PredictiveHierarchy[{' → '.join(level_strs)}]"


class PredictiveCorticalColumn:
    """Cortical column with predictive coding microcircuit.
    
    Implements the canonical microcircuit for predictive coding
    as described by Bastos et al. (2012):
    
    - L4: Receives feedforward input, computes errors
    - L2/3: Generates predictions, sends feedback
    - L5: Output layer, motor predictions
    - L6: Modulates precision
    
    Args:
        network: PyMonntorch Network.
        column_size: Base size for column layers.
        config: PredictiveCodingConfig.
        name: Column identifier.
    """
    
    def __init__(
        self,
        network: Network,
        column_size: int = 100,
        config: Optional[PredictiveCodingConfig] = None,
        name: str = "column",
    ):
        self.network = network
        self.column_size = column_size
        self.config = config or PredictiveCodingConfig()
        self.name = name
        
        # Layer sizes based on cortical proportions
        self.layer_sizes = {
            "L4": column_size,  # Granular layer (input)
            "L23": int(column_size * 1.5),  # Supragranular (prediction)
            "L5": int(column_size * 0.8),  # Infragranular (output)
            "L6": int(column_size * 0.5),  # Precision modulation
        }
        
        # Layers
        self.L4: Optional[CorticalLayer] = None
        self.L23: Optional[CorticalLayer] = None
        self.L5: Optional[CorticalLayer] = None
        self.L6: Optional[CorticalLayer] = None
        
        # Connections
        self.synapses: Dict[str, Synapsis] = {}
        
        # Build column
        self._build()
    
    def _build(self) -> None:
        """Build the cortical column microcircuit."""
        cfg = self.config
        
        # L4: Error computation layer (receives FF input)
        self.L4 = CorticalLayer(
            net=self.network,
            size=self.layer_sizes["L4"],
            behavior={
                100: ErrorUnit(config=cfg, precision_weighted=True),
                110: PrecisionWeighting(config=cfg),
                120: FreeEnergyMinimization(config=cfg),
            },
            tag=f"{self.name}_L4",
        )
        
        # L2/3: Prediction generation layer
        self.L23 = CorticalLayer(
            net=self.network,
            size=self.layer_sizes["L23"],
            behavior={
                100: PredictionUnit(config=cfg, use_temporal=True),
                110: ErrorUnit(config=cfg),
                120: PredictiveCodingLearning(config=cfg),
            },
            tag=f"{self.name}_L23",
        )
        
        # L5: Output/motor prediction layer
        self.L5 = CorticalLayer(
            net=self.network,
            size=self.layer_sizes["L5"],
            behavior={
                100: PredictionUnit(config=cfg),
                110: PredictiveCodingLearning(config=cfg),
            },
            tag=f"{self.name}_L5",
        )
        
        # L6: Precision modulation layer
        self.L6 = CorticalLayer(
            net=self.network,
            size=self.layer_sizes["L6"],
            behavior={
                100: PrecisionWeighting(config=cfg, learnable=True),
            },
            tag=f"{self.name}_L6",
        )
        
        # Create canonical connections
        self._create_connections()
    
    def _create_connections(self) -> None:
        """Create canonical microcircuit connections."""
        cfg = self.config
        
        # L4 → L2/3: Feedforward (error propagation)
        self.synapses["L4_to_L23"] = Synapsis(
            net=self.network,
            src=self.L4,
            dst=self.L23,
            behavior={
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                110: FeedforwardErrorSynapse(config=cfg),
                120: PredictiveSynapseLearning(config=cfg),
            },
            tag=f"{self.name}_L4_L23",
        )
        
        # L2/3 → L4: Feedback (predictions)
        self.synapses["L23_to_L4"] = Synapsis(
            net=self.network,
            src=self.L23,
            dst=self.L4,
            behavior={
                100: PredictiveSynapseInit(config=cfg, connection_type="feedback"),
                110: FeedbackPredictionSynapse(config=cfg),
                120: PredictiveSynapseLearning(config=cfg),
            },
            tag=f"{self.name}_L23_L4",
        )
        
        # L2/3 → L5: Output generation
        self.synapses["L23_to_L5"] = Synapsis(
            net=self.network,
            src=self.L23,
            dst=self.L5,
            behavior={
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                110: FeedforwardErrorSynapse(config=cfg),
            },
            tag=f"{self.name}_L23_L5",
        )
        
        # L6 → L4: Precision modulation
        self.synapses["L6_to_L4"] = Synapsis(
            net=self.network,
            src=self.L6,
            dst=self.L4,
            behavior={
                100: PredictiveSynapseInit(config=cfg, connection_type="lateral"),
            },
            tag=f"{self.name}_L6_L4",
        )
    
    def set_input(self, data: Tensor) -> None:
        """Set feedforward input to L4.
        
        Args:
            data: Input tensor.
        """
        if self.L4 is not None:
            self.L4.activity = data.to(
                device=self.network.device,
                dtype=self.network.def_dtype,
            )
    
    def get_prediction(self) -> Tensor:
        """Get current prediction from L2/3."""
        if self.L23 is not None and hasattr(self.L23, "prediction"):
            return self.L23.prediction
        return torch.zeros(self.layer_sizes["L23"])
    
    def get_output(self) -> Tensor:
        """Get output from L5."""
        if self.L5 is not None:
            return self.L5.activity
        return torch.zeros(self.layer_sizes["L5"])
    
    def get_error(self) -> Tensor:
        """Get prediction error from L4."""
        if self.L4 is not None and hasattr(self.L4, "prediction_error"):
            return self.L4.prediction_error
        return torch.zeros(self.layer_sizes["L4"])
    
    def get_free_energy(self) -> float:
        """Get free energy of the column."""
        if self.L4 is not None and hasattr(self.L4, "free_energy"):
            return self.L4.free_energy.item()
        return 0.0
    
    def __repr__(self) -> str:
        return f"PredictiveCorticalColumn('{self.name}', size={self.column_size})"
