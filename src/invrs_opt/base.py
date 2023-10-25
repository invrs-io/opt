"""Base objects and datatypes for optimization algorithms.

Copyright (c) 2023 Martin F. Schubert
"""

import dataclasses
from typing import Any, Callable

from totypes import json_utils

PyTree = Any


@dataclasses.dataclass
class Optimizer:
    """Stores the `(init, params, update)` function triple for an optimizer."""

    init: Callable[[PyTree], PyTree]
    params: Callable[[PyTree], PyTree]
    update: Callable[[PyTree, float, PyTree, PyTree], PyTree]


# Additional custom types and prefixes used for serializing optimizer state.
CUSTOM_TYPES_AND_PREFIXES = ()


def serialize(tree: PyTree) -> str:
    """Serializes a pytree into a string."""
    return json_utils.json_from_pytree(
        tree,
        extra_custom_types_and_prefixes=CUSTOM_TYPES_AND_PREFIXES,
    )


def deserialize(serialized: str) -> PyTree:
    """Restores a pytree from a string."""
    return json_utils.pytree_from_json(
        serialized,
        extra_custom_types_and_prefixes=CUSTOM_TYPES_AND_PREFIXES,
    )
