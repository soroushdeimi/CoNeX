"""Utilities for CoNeX neural networks."""

from .replication import *
from .precision import (
    PrecisionMode,
    PrecisionConfig,
    MixedPrecisionManager,
    PrecisionContext,
    convert_to_precision,
    check_precision_support,
    get_optimal_precision,
)

__all__ = [
    # Replication
    "replicate",
    "save_structure",
    "create_structure_from_dict",
    "save_structure_dict_to_json",
    "load_structure_dict_from_json",
    # Precision
    "PrecisionMode",
    "PrecisionConfig",
    "MixedPrecisionManager",
    "PrecisionContext",
    "convert_to_precision",
    "check_precision_support",
    "get_optimal_precision",
]
