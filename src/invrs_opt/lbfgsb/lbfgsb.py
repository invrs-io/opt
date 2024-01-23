"""Defines a jax-style wrapper for scipy's L-BFGS-B algorithm.

Copyright (c) 2023 The INVRS-IO authors.
"""

import copy
import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from jax import flatten_util
from jax import tree_util
from scipy.optimize._lbfgsb_py import (  # type: ignore[import-untyped]
    _lbfgsb as scipy_lbfgsb,
)

from invrs_opt.lbfgsb import transform
from invrs_opt import base
from totypes import types

NDArray = onp.ndarray[Any, Any]
PyTree = Any
ElementwiseBound = Union[NDArray, Sequence[Optional[float]]]
LbfgsbState = Tuple[PyTree, Dict[str, NDArray]]


# Task message prefixes for the underlying L-BFGS-B implementation.
TASK_START = b"START"
TASK_FG = b"FG"

# Parameters which configure the state update step.
UPDATE_IPRINT = -1
UPDATE_PGTOL = 0.0
UPDATE_FACTR = 0.0

# Maximum value for the `maxcor` parameter in the L-BFGS-B scheme.
MAXCOR_MAX_VALUE = 100
MAXCOR_DEFAULT = 20
LINE_SEARCH_MAX_STEPS_DEFAULT = 100

# Maps bound scenarios to integers.
BOUNDS_MAP: Dict[Tuple[bool, bool], int] = {
    (True, True): 0,  # Both upper and lower bound are `None`.
    (False, True): 1,  # Only upper bound is `None`.
    (False, False): 2,  # Neither of the bounds are `None`.
    (True, False): 3,  # Only the lower bound is `None`.
}

FORTRAN_INT = scipy_lbfgsb.types.intvar.dtype


def lbfgsb(
    maxcor: int = MAXCOR_DEFAULT,
    line_search_max_steps: int = LINE_SEARCH_MAX_STEPS_DEFAULT,
) -> base.Optimizer:
    """Return an optimizer implementing the standard L-BFGS-B algorithm.

    This optimizer wraps scipy's implementation of the algorithm, and provides
    a jax-style API to the scheme. The optimizer works with custom types such
    as the `BoundedArray` to constrain the optimization variable.

    Example usage is as follows:

        def fn(x):
            leaves_sum_sq = [jnp.sum(y)**2 for y in tree_util.tree_leaves(x)]
            return jnp.sum(jnp.asarray(leaves_sum_sq))

        x0 = {
            "a": jnp.ones((3,)),
            "b": BoundedArray(
                value=-jnp.ones((2, 5)),
                lower_bound=-5,
                upper_bound=5,
            ),
        }
        opt = lbfgsb(maxcor=20, line_search_max_steps=100)
        state = opt.init(x0)
        for _ in range(10):
            x = opt.params(state)
            value, grad = jax.value_and_grad(fn)(x)
            state = opt.update(grad, value, state)

    While the algorithm can work with pytrees of jax arrays, numpy arrays can
    also be used. Thus, e.g. the optimizer can directly be used with autograd.

    Args:
        maxcor: The maximum number of variable metric corrections used to define
            the limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.

    Returns:
        The `base.Optimizer`.
    """
    return transformed_lbfgsb(
        maxcor=maxcor,
        line_search_max_steps=line_search_max_steps,
        transform_fn=lambda x: x,
    )


def density_lbfgsb(
    beta: float,
    maxcor: int = MAXCOR_DEFAULT,
    line_search_max_steps: int = LINE_SEARCH_MAX_STEPS_DEFAULT,
) -> base.Optimizer:
    """Return an L-BFGS-B optimizer with additional transforms for density arrays.

    Parameters that are of type `DensityArray2D` are represented as latent parameters
    that are transformed (in the case where lower and upper bounds are `(-1, 1)`) by,

        transformed = tanh(beta * conv(density.array, gaussian_kernel)) / tanh(beta)

    where the kernel has a full-width at half-maximum determined by the minimum width
    and spacing parameters of the `DensityArray2D`. Where the bounds differ, the
    density is scaled before the transform is applied, and then unscaled afterwards.

    Args:
        beta: Determines the steepness of the thresholding.
        maxcor: The maximum number of variable metric corrections used to define
            the limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.

    Returns:
        The `base.Optimizer`.
    """

    def transform_fn(tree: PyTree) -> PyTree:
        return tree_util.tree_map(
            lambda x: (
                transform_density(x) if isinstance(x, types.Density2DArray) else x
            ),
            tree,
            is_leaf=lambda x: isinstance(x, types.CUSTOM_TYPES),
        )

    def transform_density(density: types.Density2DArray) -> types.Density2DArray:
        transformed = types.symmetrize_density(density)
        transformed = transform.density_gaussian_filter_and_tanh(transformed, beta=beta)
        # Scale to ensure that the full valid range of the density array is reachable.
        mid_value = (density.lower_bound + density.upper_bound) / 2
        transformed = tree_util.tree_map(
            lambda array: mid_value + (array - mid_value) / jnp.tanh(beta), transformed
        )
        return transform.apply_fixed_pixels(transformed)

    return transformed_lbfgsb(
        maxcor=maxcor,
        line_search_max_steps=line_search_max_steps,
        transform_fn=transform_fn,
    )


def transformed_lbfgsb(
    maxcor: int,
    line_search_max_steps: int,
    transform_fn: Callable[[PyTree], PyTree],
) -> base.Optimizer:
    """Construct an latent parameter L-BFGS-B optimizer.

    The optimized parameters are termed latent parameters, from which the
    actual parameters returned by the optimizer are obtained using the
    `transform_fn`. In the simple case where this is just `lambda x: x` (i.e.
    the identity), this is equivalent to the standard L-BFGS-B algorithm.

    Args:
        maxcor: The maximum number of variable metric corrections used to define
            the limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.
        transform_fn: Function which transforms the internal latent parameters to
            the parameters returned by the optimizer.

    Returns:
        The `base.Optimizer`.
    """
    if not isinstance(maxcor, int) or maxcor < 1 or maxcor > MAXCOR_MAX_VALUE:
        raise ValueError(
            f"`maxcor` must be greater than 0 and less than "
            f"{MAXCOR_MAX_VALUE}, but got {maxcor}"
        )

    if not isinstance(line_search_max_steps, int) or line_search_max_steps < 1:
        raise ValueError(
            f"`line_search_max_steps` must be greater than 0 but got "
            f"{line_search_max_steps}"
        )

    def init_fn(params: PyTree) -> LbfgsbState:
        """Initializes the optimization state."""
        lower_bound = types.extract_lower_bound(params)
        upper_bound = types.extract_upper_bound(params)
        scipy_lbfgsb_state = ScipyLbfgsbState.init(
            x0=_to_numpy(params),
            lower_bound=_bound_for_params(lower_bound, params),
            upper_bound=_bound_for_params(upper_bound, params),
            maxcor=maxcor,
            line_search_max_steps=line_search_max_steps,
        )
        latent_params = _to_pytree(scipy_lbfgsb_state.x, params)
        params = transform_fn(latent_params)
        return (params, dataclasses.asdict(scipy_lbfgsb_state))

    def params_fn(state: LbfgsbState) -> PyTree:
        """Returns the parameters for the given `state`."""
        params, _ = state
        return params

    def update_fn(
        *,
        grad: PyTree,
        value: float,
        params: PyTree,
        state: LbfgsbState,
    ) -> LbfgsbState:
        """Updates the state."""
        del params
        params, lbfgsb_state_dict = state
        # Avoid in-place updates.
        lbfgsb_state_dict = copy.deepcopy(lbfgsb_state_dict)
        scipy_lbfgsb_state = ScipyLbfgsbState(
            **lbfgsb_state_dict  # type: ignore[arg-type]
        )

        latent_params = _to_pytree(scipy_lbfgsb_state.x, params)
        _, vjp_fn = jax.vjp(transform_fn, latent_params)
        (latent_grad,) = vjp_fn(grad)

        assert onp.size(value) == 1
        scipy_lbfgsb_state.update(grad=_to_numpy(latent_grad), value=onp.asarray(value))
        latent_params = _to_pytree(scipy_lbfgsb_state.x, params)
        params = transform_fn(latent_params)
        return (params, dataclasses.asdict(scipy_lbfgsb_state))

    return base.Optimizer(
        init=init_fn,
        params=params_fn,
        update=update_fn,
    )


# ------------------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------------------


def _to_numpy(params: PyTree) -> NDArray:
    """Flattens a `params` pytree into a single rank-1 numpy array."""
    x, _ = flatten_util.ravel_pytree(params)  # type: ignore[no-untyped-call]
    return onp.asarray(x, dtype=onp.float64)


def _to_pytree(x_flat: NDArray, params: PyTree) -> PyTree:
    """Restores a pytree from a flat numpy array using the structure of `params`.

    Note that the returned pytree includes jax array leaves.

    Args:
        x_flat: The rank-1 numpy array to be restored.
        params: A pytree of parameters whose structure is replicated in the restored
            pytree.

    Returns:
        The restored pytree, with jax array leaves.
    """
    _, unflatten_fn = flatten_util.ravel_pytree(params)  # type: ignore[no-untyped-call]
    return unflatten_fn(jnp.asarray(x_flat, dtype=float))


def _bound_for_params(bound: PyTree, params: PyTree) -> ElementwiseBound:
    """Generates a bound vector for the `params`.

    The `bound` can be specified in various ways; it may be `None` or a scalar,
    which then applies to all arrays in `params`. It may be a pytree with
    structure matching that of `params`, where each leaf is either `None`, a
    scalar, or an array matching the shape of the corresponding leaf in `params`.

    The returned bound is a flat array suitable for use with `ScipyLbfgsbState`.

    Args:
        bound: The pytree of bounds.
        params: The pytree of parameters.

    Returns:
        The flat elementwise bound.
    """

    if bound is None or onp.isscalar(bound):
        bound = tree_util.tree_map(
            lambda _: bound,
            params,
            is_leaf=lambda x: isinstance(x, types.CUSTOM_TYPES),
        )

    bound_leaves, bound_treedef = tree_util.tree_flatten(
        bound, is_leaf=lambda x: x is None
    )
    params_leaves = tree_util.tree_leaves(params, is_leaf=lambda x: x is None)

    # `bound` should be a pytree of arrays or `None`, while `params` may
    # include custom pytree nodes. Convert the custom nodes into standard
    # types to facilitate validation that the tree structures match.
    params_treedef = tree_util.tree_structure(
        tree_util.tree_map(
            lambda x: 0.0,
            tree=params,
            is_leaf=lambda x: x is None or isinstance(x, types.CUSTOM_TYPES),
        )
    )
    if bound_treedef != params_treedef:  # type: ignore[operator]
        raise ValueError(
            f"Tree structure of `bound` and `params` must match, but got "
            f"{bound_treedef} and {params_treedef}, respectively."
        )

    bound_flat = []
    for b, p in zip(bound_leaves, params_leaves):
        if p is None:
            continue
        if b is None or onp.isscalar(b) or onp.shape(b) == ():
            bound_flat += [b] * onp.size(p)
        else:
            if b.shape != p.shape:
                raise ValueError(
                    f"`bound` must be `None`, a scalar, or have shape matching "
                    f"`params`, but got shape {b.shape} when params has shape "
                    f"{p.shape}."
                )
            bound_flat += b.flatten().tolist()

    return bound_flat


# ------------------------------------------------------------------------------
# Wrapper for scipy's L-BFGS-B implementation.
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class ScipyLbfgsbState:
    """Stores the state of a scipy L-BFGS-B minimization.

    This class enables optimization with a more functional style, giving the user
    control over the optimization loop. Example usage is as follows:

        value_fn = lambda x: onp.sum(x**2)
        grad_fn = lambda x: 2 * x

        x0 = onp.asarray([0.1, 0.2, 0.3])
        lb = [None, -1, 0.1]
        ub = [None, None, None]
        state = ScipyLbfgsbState.init(
            x0=x0, lower_bound=lb, upper_bound=ub, maxcor=20
        )

        for _ in range(10):
            value = value_fn(state.x)
            grad = grad_fn(state.x)
            state.update(grad, value)

    This example converges with `state.x` equal to `(0, 0, 0.1)` and value equal
    to `0.01`.

    Attributes:
        x: The current solution vector.
    """

    x: NDArray
    # Private attributes correspond to internal variables in the `scipy.optimize.
    # lbfgsb._minimize_lbfgsb` function.
    _maxcor: int
    _line_search_max_steps: int
    _wa: NDArray
    _iwa: NDArray
    _task: NDArray
    _csave: NDArray
    _lsave: NDArray
    _isave: NDArray
    _dsave: NDArray
    _lower_bound: NDArray
    _upper_bound: NDArray
    _bound_type: NDArray

    def __post_init__(self) -> None:
        """Validates the datatypes for all state attributes."""
        _validate_array_dtype(self.x, onp.float64)
        _validate_array_dtype(self._wa, onp.float64)
        _validate_array_dtype(self._iwa, FORTRAN_INT)
        _validate_array_dtype(self._task, "S60")
        _validate_array_dtype(self._csave, "S60")
        _validate_array_dtype(self._lsave, FORTRAN_INT)
        _validate_array_dtype(self._isave, FORTRAN_INT)
        _validate_array_dtype(self._dsave, onp.float64)
        _validate_array_dtype(self._lower_bound, onp.float64)
        _validate_array_dtype(self._upper_bound, onp.float64)
        _validate_array_dtype(self._bound_type, int)

    @classmethod
    def init(
        cls,
        x0: NDArray,
        lower_bound: ElementwiseBound,
        upper_bound: ElementwiseBound,
        maxcor: int,
        line_search_max_steps: int,
    ) -> "ScipyLbfgsbState":
        """Initializes the `ScipyLbfgsbState` for `x0`.

        Args:
            x0: Array giving the initial solution vector.
            lower_bound: Array giving the elementwise optional lower bound.
            upper_bound: Array giving the elementwise optional upper bound.
            maxcor: The maximum number of variable metric corrections used to define
                the limited memory matrix, in the L-BFGS-B scheme.
            line_search_max_steps: The maximum number of steps in the line search.

        Returns:
            The `ScipyLbfgsbState`.
        """
        x0 = onp.asarray(x0)
        if x0.ndim > 1:
            raise ValueError(f"`x0` must be rank-1 but got shape {x0.shape}.")
        lower_bound = onp.asarray(lower_bound)
        upper_bound = onp.asarray(upper_bound)
        if x0.shape != lower_bound.shape or x0.shape != upper_bound.shape:
            raise ValueError(
                f"`x0`, `lower_bound`, and `upper_bound` must have matching "
                f"shape but got shapes {x0.shape}, {lower_bound.shape}, and "
                f"{upper_bound.shape}, respectively."
            )
        if maxcor < 1:
            raise ValueError(f"`maxcor` must be positive but got {maxcor}.")

        n = x0.size
        lower_bound_array, upper_bound_array, bound_type = _configure_bounds(
            lower_bound, upper_bound
        )
        task = onp.zeros(1, "S60")
        task[:] = TASK_START

        # See initialization of internal variables in the `lbfgsb._minimize_lbfgsb`
        # function.
        wa_shape = 2 * maxcor * n + 5 * n + 11 * maxcor**2 + 8 * maxcor
        state = ScipyLbfgsbState(
            x=onp.array(x0, onp.float64),
            _maxcor=maxcor,
            _line_search_max_steps=line_search_max_steps,
            _wa=onp.zeros(wa_shape, onp.float64),
            _iwa=onp.zeros(3 * n, FORTRAN_INT),
            _task=task,
            _csave=onp.zeros(1, "S60"),
            _lsave=onp.zeros(4, FORTRAN_INT),
            _isave=onp.zeros(44, FORTRAN_INT),
            _dsave=onp.zeros(29, onp.float64),
            _lower_bound=lower_bound_array,
            _upper_bound=upper_bound_array,
            _bound_type=bound_type,
        )
        # The initial state requires an update with zero value and gradient. This
        # is because the initial task is "START", which does not actually require
        # value and gradient evaluation.
        state.update(onp.zeros(x0.shape, onp.float64), onp.zeros((), onp.float64))
        return state

    def update(
        self,
        grad: NDArray,
        value: NDArray,
    ) -> None:
        """Performs an in-place update of the `ScipyLbfgsbState`.

        Args:
            grad: The function gradient for the current `x`.
            value: The scalar function value for the current `x`.
        """
        if grad.shape != self.x.shape:
            raise ValueError(
                f"`grad` must have the same shape as attribute `x`, but got shapes "
                f"{grad.shape} and {self.x.shape}, respectively."
            )
        if value.shape != ():
            raise ValueError(f"`value` must be a scalar but got shape {value.shape}.")

        # The `setulb` function will sometimes return with a task that does not
        # require a value and gradient evaluation. In this case we simply call it
        # again, advancing past such "dummy" steps.
        for _ in range(3):
            scipy_lbfgsb.setulb(
                m=self._maxcor,
                x=self.x,
                l=self._lower_bound,
                u=self._upper_bound,
                nbd=self._bound_type,
                f=value,
                g=grad,
                factr=UPDATE_FACTR,
                pgtol=UPDATE_PGTOL,
                wa=self._wa,
                iwa=self._iwa,
                task=self._task,
                iprint=UPDATE_IPRINT,
                csave=self._csave,
                lsave=self._lsave,
                isave=self._isave,
                dsave=self._dsave,
                maxls=self._line_search_max_steps,
            )
            task_str = self._task.tobytes()
            if task_str.startswith(TASK_FG):
                break


def _validate_array_dtype(x: NDArray, dtype: Union[type, str]) -> None:
    """Validates that `x` is an array with the specified `dtype`."""
    if not isinstance(x, onp.ndarray):
        raise ValueError(f"`x` must be an `onp.ndarray` but got {type(x)}")
    if x.dtype != dtype:
        raise ValueError(f"`x` must have dtype {dtype} but got {x.dtype}")


def _configure_bounds(
    lower_bound: ElementwiseBound,
    upper_bound: ElementwiseBound,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Configures the bounds for an L-BFGS-B optimization."""
    bound_type = [
        BOUNDS_MAP[(lower is None, upper is None)]
        for lower, upper in zip(lower_bound, upper_bound)
    ]
    lower_bound_array = [0.0 if x is None else x for x in lower_bound]
    upper_bound_array = [0.0 if x is None else x for x in upper_bound]
    return (
        onp.asarray(lower_bound_array, onp.float64),
        onp.asarray(upper_bound_array, onp.float64),
        onp.asarray(bound_type),
    )
