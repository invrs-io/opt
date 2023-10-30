"""Base objects and datatypes for optimization algorithms.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
from typing import Any, Protocol

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
