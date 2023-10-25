"""Base objects and datatypes for optimization algorithms.

Copyright (c) 2023 Martin F. Schubert
"""

import dataclasses
from typing import Any, Protocol

from totypes import json_utils

PyTree = Any


class InitFn(Protocol):
    """Callable which initializes an optimizer state."""

    def __call__(self, params: PyTree) -> PyTree:
        ...


class ParamsFn(Protocol):
    """Callable which returns the parameters for an optimizer state."""

    def __call__(self, state: PyTree) -> PyTree:
        ...


class UpdateFn(Protocol):
    """Callable which updates an optimizer state."""

    def __call__(
        self,
        *,
        grad: PyTree,
        value: float,
        params: PyTree,
        state: PyTree,
    ) -> PyTree:
        ...


@dataclasses.dataclass
class Optimizer:
    """Stores the `(init, params, update)` function triple for an optimizer."""

    init: InitFn
    params: ParamsFn
    update: UpdateFn


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
