"""Base objects and datatypes for optimization algorithms.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
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


# TODO: consider programatically registering all optax states here.
json_utils.register_custom_type(optax.EmptyState)
json_utils.register_custom_type(optax.ScaleByAdamState)
