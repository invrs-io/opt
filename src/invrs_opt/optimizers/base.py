"""Base objects and datatypes for optimization algorithms.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import inspect
from typing import Any, Protocol

import optax  # type: ignore[import-untyped]
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


# Register all optax state types for serialization.
optax_types = {}
for name, obj in inspect.getmembers(optax):
    if name.endswith("State") and isinstance(obj, type):
        optax_types[obj] = True

for obj in optax_types.keys():
    json_utils.register_custom_type(obj)
