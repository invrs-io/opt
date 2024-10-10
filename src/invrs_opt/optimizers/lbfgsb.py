"""Defines a jax-style wrapper for scipy's L-BFGS-B algorithm.

Copyright (c) 2023 The INVRS-IO authors.
"""

import copy
import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
import optax  # type: ignore[import-untyped]
from jax import flatten_util, tree_util
from scipy.optimize._lbfgsb_py import (  # type: ignore[import-untyped]
    _lbfgsb as scipy_lbfgsb,
)
from totypes import types

from invrs_opt.optimizers import base
from invrs_opt.parameterization import (
    base as param_base,
    filter_project,
    gaussian_levelset,
    pixel,
)

NDArray = onp.ndarray[Any, Any]
PyTree = Any
ElementwiseBound = Union[NDArray, Sequence[Optional[float]]]
JaxLbfgsbDict = Dict[str, jnp.ndarray]
LbfgsbState = Tuple[int, PyTree, PyTree, JaxLbfgsbDict]


# Task message prefixes for the underlying L-BFGS-B implementation.
TASK_START = b"START"
TASK_FG = b"FG"
TASK_CONVERGED = b"CONVERGENCE"

UPDATE_IPRINT = -1

# Maximum value for the `maxcor` parameter in the L-BFGS-B scheme.
MAXCOR_MAX_VALUE = 100
DEFAULT_MAXCOR = 20
DEFAULT_LINE_SEARCH_MAX_STEPS = 100
DEFAULT_FTOL = 0.0
DEFAULT_GTOL = 0.0

# Maps bound scenarios to integers.
BOUNDS_MAP: Dict[Tuple[bool, bool], int] = {
    (True, True): 0,  # Both upper and lower bound are `None`.
    (False, True): 1,  # Only upper bound is `None`.
    (False, False): 2,  # Neither of the bounds are `None`.
    (True, False): 3,  # Only the lower bound is `None`.
}

FORTRAN_INT = scipy_lbfgsb.types.intvar.dtype


def lbfgsb(
    *,
    maxcor: int = DEFAULT_MAXCOR,
    line_search_max_steps: int = DEFAULT_LINE_SEARCH_MAX_STEPS,
    ftol: float = DEFAULT_FTOL,
    gtol: float = DEFAULT_GTOL,
) -> base.Optimizer:
    """Optimizer implementing the standard L-BFGS-B algorithm.

    The standard L-BFGS-B algorithm uses the direct pixel parameterization for density
    arrays, which simply enforces that values are between the declared upper and lower
    bounds of the density.

    When an optimization is determined to have converged (by `ftol` or `gtol` criteria)
    the optimizer `params` function will simply return the optimal parameters. The
    convergence can be queried by `is_converged(state)`.

    Args:
        maxcor: The maximum number of variable metric corrections used to define the
            limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.
        ftol: Convergence criteria based on function values. See scipy documentation
            for details.
        gtol: Convergence criteria based on gradient.

    Returns:
        The `Optimizer` implementing the L-BFGS-B optimizer.
    """
    return parameterized_lbfgsb(
        density_parameterization=None,
        penalty=0.0,
        maxcor=maxcor,
        line_search_max_steps=line_search_max_steps,
        ftol=ftol,
        gtol=gtol,
    )


def density_lbfgsb(
    *,
    beta: float,
    maxcor: int = DEFAULT_MAXCOR,
    line_search_max_steps: int = DEFAULT_LINE_SEARCH_MAX_STEPS,
    ftol: float = DEFAULT_FTOL,
    gtol: float = DEFAULT_GTOL,
) -> base.Optimizer:
    """Optimizer using L-BFGS-B algorithm with filter-project density parameterization.

    In the filter-project density parameterization, the optimization variable
    associated with a density array is a latent density array; the density is obtained
    by convolving (i.e. "filtering") the latent density with a Gaussian kernel having
    full-width at half-maximum equal to the length scale (the mean of declared minimum
    width and minimum spacing). Then, a tanh nonlinearity is used as a smooth threshold
    operation ("projection").

    When an optimization is determined to have converged (by `ftol` or `gtol` criteria)
    the optimizer `params` function will simply return the optimal parameters. The
    convergence can be queried by `is_converged(state)`.

    Args:
        beta: Determines the sharpness of the thresholding operation.
        maxcor: The maximum number of variable metric corrections used to define the
            limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.
        ftol: Convergence criteria based on function values. See scipy documentation
            for details.
        gtol: Convergence criteria based on gradient.

    Returns:
        The `Optimizer` implementing the L-BFGS-B optimizer.
    """
    return parameterized_lbfgsb(
        density_parameterization=filter_project.filter_project(beta=beta),
        penalty=0.0,
        maxcor=maxcor,
        line_search_max_steps=line_search_max_steps,
        ftol=ftol,
        gtol=gtol,
    )


def levelset_lbfgsb(
    *,
    penalty: float,
    length_scale_spacing_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_SPACING_FACTOR
    ),
    length_scale_fwhm_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_FWHM_FACTOR
    ),
    length_scale_constraint_factor: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_FACTOR
    ),
    smoothing_factor: int = gaussian_levelset.DEFAULT_SMOOTHING_FACTOR,
    length_scale_constraint_beta: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_BETA
    ),
    length_scale_constraint_weight: float = (
        gaussian_levelset.DEFAULT_LENGTH_SCALE_CONSTRAINT_WEIGHT
    ),
    curvature_constraint_weight: float = (
        gaussian_levelset.DEFAULT_CURVATURE_CONSTRAINT_WEIGHT
    ),
    fixed_pixel_constraint_weight: float = (
        gaussian_levelset.DEFAULT_FIXED_PIXEL_CONSTRAINT_WEIGHT
    ),
    init_optimizer: optax.GradientTransformation = (
        gaussian_levelset.DEFAULT_INIT_OPTIMIZER
    ),
    init_steps: int = gaussian_levelset.DEFAULT_INIT_STEPS,
    maxcor: int = DEFAULT_MAXCOR,
    line_search_max_steps: int = DEFAULT_LINE_SEARCH_MAX_STEPS,
    ftol: float = DEFAULT_FTOL,
    gtol: float = DEFAULT_GTOL,
) -> base.Optimizer:
    """Optimizer using L-BFGS-B algorithm with levelset density parameterization.

    In the levelset parameterization, the optimization variable associated with a
    density array is an array giving the amplitudes of Gaussian radial basis functions
    that represent a levelset function over the domain of the density. In the levelset
    parameterization, gradients are nonzero only at the edges of features, and in
    general the topology of a solution does not change during the course of
    optimization.

    The spacing and full-width at half-maximum of the Gaussian basis functions gives
    some amount of control over length scales. In addition, constraints associated with
    length scale, radius of curvature, and deviation from fixed pixels are
    automatically computed and penalized with a weight given by `penalty`. In general,
    this helps ensure that features in an optimized density array violate the specified
    constraints to a lesser degree. The constraints are based on "Analytical level set
    fabrication constraints for inverse design," by D. Vercruysse et al. (2019).

    When an optimization is determined to have converged (by `ftol` or `gtol` criteria)
    the optimizer `params` function will simply return the optimal parameters. The
    convergence can be queried by `is_converged(state)`.

    Args:
        penalty: The weight of the fabrication penalty, which combines length scale,
            curvature, and fixed pixel constraints.
        length_scale_spacing_factor: The number of levelset control points per unit of
            minimum length scale (mean of density minimum width and minimum spacing).
        length_scale_fwhm_factor: The ratio of Gaussian full-width at half-maximum to
            the minimum length scale.
        length_scale_constraint_factor: Multiplies the target length scale in the
            levelset constraints. A value greater than 1 is pessimistic and drives the
            solution to have a larger length scale (relative to smaller values).
        smoothing_factor: For values greater than 1, the density is initially computed
            at higher resolution and then downsampled, yielding smoother geometries.
        length_scale_constraint_beta: Controls relaxation of the length scale
            constraint near the zero level.
        length_scale_constraint_weight: The weight of the length scale constraint in
            the overall fabrication constraint peenalty.
        curvature_constraint_weight: The weight of the curvature constraint.
        fixed_pixel_constraint_weight: The weight of the fixed pixel constraint.
        init_optimizer: The optimizer used in the initialization of the levelset
            parameterization. At initialization, the latent parameters are optimized so
            that the initial parameters match the binarized initial density.
        init_steps: The number of optimization steps used in the initialization.
        maxcor: The maximum number of variable metric corrections used to define the
            limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.
        ftol: Convergence criteria based on function values. See scipy documentation
            for details.
        gtol: Convergence criteria based on gradient.

    Returns:
        The `Optimizer` implementing the L-BFGS-B optimizer.
    """
    return parameterized_lbfgsb(
        density_parameterization=gaussian_levelset.gaussian_levelset(
            length_scale_spacing_factor=length_scale_spacing_factor,
            length_scale_fwhm_factor=length_scale_fwhm_factor,
            length_scale_constraint_factor=length_scale_constraint_factor,
            smoothing_factor=smoothing_factor,
            length_scale_constraint_beta=length_scale_constraint_beta,
            length_scale_constraint_weight=length_scale_constraint_weight,
            curvature_constraint_weight=curvature_constraint_weight,
            fixed_pixel_constraint_weight=fixed_pixel_constraint_weight,
            init_optimizer=init_optimizer,
            init_steps=init_steps,
        ),
        penalty=penalty,
        maxcor=maxcor,
        line_search_max_steps=line_search_max_steps,
        ftol=ftol,
        gtol=gtol,
    )


# -----------------------------------------------------------------------------
# Base parameterized L-BFGS-B optimizer.
# -----------------------------------------------------------------------------


def parameterized_lbfgsb(
    density_parameterization: Optional[param_base.Density2DParameterization],
    penalty: float,
    maxcor: int = DEFAULT_MAXCOR,
    line_search_max_steps: int = DEFAULT_LINE_SEARCH_MAX_STEPS,
    ftol: float = DEFAULT_FTOL,
    gtol: float = DEFAULT_GTOL,
) -> base.Optimizer:
    """Optimizer using L-BFGS-B optimizer with specified density parameterization.

    This optimizer wraps scipy's implementation of the algorithm, and provides
    a jax-style API to the scheme. The optimizer works with custom types such
    as the `BoundedArray` to constrain the optimization variable.

    Args:
        density_parameterization: The parameterization to be used, or `None`. When no
            parameterization is given, the direct pixel parameterization is used for
            density arrays.
        penalty: The weight of the scalar penalty formed from the constraints of the
            parameterization.
        maxcor: The maximum number of variable metric corrections used to define the
            limited memory matrix, in the L-BFGS-B scheme.
        line_search_max_steps: The maximum number of steps in the line search.
        ftol: Convergence criteria based on function values. See scipy documentation
            for details.
        gtol: Convergence criteria based on gradient.

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

    if density_parameterization is None:
        density_parameterization = pixel.pixel()

    def init_fn(params: PyTree) -> LbfgsbState:
        """Initializes the optimization state."""

        def _init_state_pure(latent_params: PyTree) -> Tuple[PyTree, JaxLbfgsbDict]:
            lower_bound = types.extract_lower_bound(latent_params)
            upper_bound = types.extract_upper_bound(latent_params)
            scipy_lbfgsb_state = ScipyLbfgsbState.init(
                x0=_to_numpy(latent_params),
                lower_bound=_bound_for_params(lower_bound, latent_params),
                upper_bound=_bound_for_params(upper_bound, latent_params),
                maxcor=maxcor,
                line_search_max_steps=line_search_max_steps,
                ftol=ftol,
                gtol=gtol,
            )
            latent_params = _to_pytree(scipy_lbfgsb_state.x, latent_params)
            return latent_params, scipy_lbfgsb_state.to_jax()

        latent_params = _init_latents(params)
        metadata, latents = param_base.partition_density_metadata(latent_params)
        latents, jax_lbfgsb_state = jax.pure_callback(
            _init_state_pure, _example_state(latents, maxcor), latents
        )
        latent_params = param_base.combine_density_metadata(metadata, latents)
        return (
            0,  # step
            _params_from_latent_params(latent_params),  # params
            latent_params,  # latent params
            jax_lbfgsb_state,  # opt state
        )

    def params_fn(state: LbfgsbState) -> PyTree:
        """Returns the parameters for the given `state`."""
        _, params, _, _ = state
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

        def _update_pure(
            flat_latent_grad: PyTree,
            value: jnp.ndarray,
            jax_lbfgsb_state: JaxLbfgsbDict,
        ) -> Tuple[PyTree, JaxLbfgsbDict]:
            assert onp.size(value) == 1
            scipy_lbfgsb_state = ScipyLbfgsbState.from_jax(jax_lbfgsb_state)
            scipy_lbfgsb_state.update(
                grad=onp.array(flat_latent_grad, dtype=onp.float64),
                value=onp.array(value, dtype=onp.float64),
            )
            flat_latent_params = jnp.asarray(scipy_lbfgsb_state.x)
            return flat_latent_params, scipy_lbfgsb_state.to_jax()

        step, _, latent_params, jax_lbfgsb_state = state
        metadata, latents = param_base.partition_density_metadata(latent_params)

        def _params_from_latents(latents: PyTree) -> PyTree:
            latent_params = param_base.combine_density_metadata(metadata, latents)
            return _params_from_latent_params(latent_params)

        def _constraint_loss_latents(latents: PyTree) -> jnp.ndarray:
            latent_params = param_base.combine_density_metadata(metadata, latents)
            return _constraint_loss(latent_params)

        _, vjp_fn = jax.vjp(_params_from_latents, latents)
        (latents_grad,) = vjp_fn(grad)

        if not (
            tree_util.tree_structure(latents_grad)
            == tree_util.tree_structure(latents)  # type: ignore[operator]
        ):
            raise ValueError(
                f"Tree structure of `latents_grad` was different than expected, got \n"
                f"{tree_util.tree_structure(latents_grad)} but expected \n"
                f"{tree_util.tree_structure(latents)}."
            )

        (
            constraint_loss_value,
            constraint_loss_grad,
        ) = jax.value_and_grad(
            _constraint_loss_latents
        )(latents)
        value += constraint_loss_value
        latents_grad = tree_util.tree_map(
            lambda a, b: a + b, latents_grad, constraint_loss_grad
        )

        flat_latents_grad, unflatten_fn = flatten_util.ravel_pytree(
            latents_grad
        )  # type: ignore[no-untyped-call]

        flat_latents, jax_lbfgsb_state = jax.pure_callback(
            _update_pure,
            (flat_latents_grad, jax_lbfgsb_state),
            flat_latents_grad,
            value,
            jax_lbfgsb_state,
        )
        latents = unflatten_fn(flat_latents)
        latent_params = param_base.combine_density_metadata(metadata, latents)
        latent_params = _update_parameterized_densities(latent_params, step)
        params = _params_from_latent_params(latent_params)
        return step + 1, params, latent_params, jax_lbfgsb_state

    # -------------------------------------------------------------------------
    # Functions related to the density parameterization.
    # -------------------------------------------------------------------------

    def _init_latents(params: PyTree) -> PyTree:
        def _leaf_init_latents(leaf: Any) -> Any:
            leaf = _clip(leaf)
            if not _is_density(leaf) or density_parameterization is None:
                return leaf
            return density_parameterization.from_density(leaf)

        return tree_util.tree_map(_leaf_init_latents, params, is_leaf=_is_custom_type)

    def _params_from_latent_params(latent_params: PyTree) -> PyTree:
        def _leaf_params_from_latents(leaf: Any) -> Any:
            if not _is_parameterized_density(leaf) or density_parameterization is None:
                return leaf
            return density_parameterization.to_density(leaf)

        return tree_util.tree_map(
            _leaf_params_from_latents,
            latent_params,
            is_leaf=_is_parameterized_density,
        )

    def _update_parameterized_densities(latent_params: PyTree, step: int) -> PyTree:
        def _update_leaf(leaf: Any) -> Any:
            if not _is_parameterized_density(leaf):
                return leaf
            return density_parameterization.update(leaf, step)

        return tree_util.tree_map(
            _update_leaf,
            latent_params,
            is_leaf=_is_parameterized_density,
        )

    # -------------------------------------------------------------------------
    # Functions related to the constraints to be minimized.
    # -------------------------------------------------------------------------

    def _constraint_loss(latent_params: PyTree) -> jnp.ndarray:
        def _constraint_loss_leaf(
            leaf: param_base.ParameterizedDensity2DArray,
        ) -> jnp.ndarray:
            constraints = density_parameterization.constraints(leaf)
            constraints = tree_util.tree_map(
                lambda x: jnp.sum(jnp.maximum(x, 0.0) ** 2),
                constraints,
            )
            return jnp.sum(jnp.asarray(constraints))

        losses = [0.0] + [
            _constraint_loss_leaf(p)
            for p in tree_util.tree_leaves(
                latent_params, is_leaf=_is_parameterized_density
            )
            if _is_parameterized_density(p)
        ]
        return penalty * jnp.sum(jnp.asarray(losses))

    return base.Optimizer(
        init=init_fn,
        params=params_fn,
        update=update_fn,
    )


def is_converged(state: LbfgsbState) -> jnp.ndarray:
    """Returns `True` if the optimization has converged."""
    return state[3]["converged"]


# ------------------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------------------


def _is_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a density array."""
    return isinstance(leaf, types.Density2DArray)


def _is_parameterized_density(leaf: Any) -> Any:
    """Return `True` if `leaf` is a parameterized density array."""
    return isinstance(leaf, param_base.ParameterizedDensity2DArray)


def _is_custom_type(leaf: Any) -> bool:
    """Return `True` if `leaf` is a recognized custom type."""
    return isinstance(leaf, (types.BoundedArray, types.Density2DArray))


def _clip(pytree: PyTree) -> PyTree:
    """Clips leaves on `pytree` to their bounds."""

    def _clip_fn(leaf: Any) -> Any:
        if not _is_custom_type(leaf):
            return leaf
        if leaf.lower_bound is None and leaf.upper_bound is None:
            return leaf
        return tree_util.tree_map(
            lambda x: jnp.clip(x, leaf.lower_bound, leaf.upper_bound), leaf
        )

    return tree_util.tree_map(_clip_fn, pytree, is_leaf=_is_custom_type)


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


def _example_state(params: PyTree, maxcor: int) -> PyTree:
    """Return an example state for the given `params` and `maxcor`."""
    params_flat, _ = flatten_util.ravel_pytree(params)  # type: ignore[no-untyped-call]
    n = params_flat.size
    float_params = tree_util.tree_map(lambda x: jnp.asarray(x, dtype=float), params)
    example_jax_lbfgsb_state = dict(
        x=jnp.zeros(n, dtype=float),
        converged=jnp.asarray(False),
        _maxcor=jnp.zeros((), dtype=int),
        _line_search_max_steps=jnp.zeros((), dtype=int),
        _ftol=jnp.zeros((), dtype=float),
        _gtol=jnp.zeros((), dtype=float),
        _wa=jnp.ones(_wa_size(n=n, maxcor=maxcor), dtype=float),
        _iwa=jnp.ones(n * 3, dtype=jnp.int32),  # Fortran int
        _task=jnp.zeros(59, dtype=int),
        _csave=jnp.zeros(59, dtype=int),
        _lsave=jnp.zeros(4, dtype=jnp.int32),  # Fortran int
        _isave=jnp.zeros(44, dtype=jnp.int32),  # Fortran int
        _dsave=jnp.zeros(29, dtype=float),
        _lower_bound=jnp.zeros(n, dtype=float),
        _upper_bound=jnp.zeros(n, dtype=float),
        _bound_type=jnp.zeros(n, dtype=int),
    )
    return float_params, example_jax_lbfgsb_state


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
    converged: NDArray
    # Private attributes correspond to internal variables in the `scipy.optimize.
    # lbfgsb._minimize_lbfgsb` function.
    _maxcor: int
    _line_search_max_steps: int
    _ftol: NDArray
    _gtol: NDArray
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

    def to_jax(self) -> Dict[str, jnp.ndarray]:
        """Generates a dictionary of jax arrays defining the state."""
        return dict(
            x=jnp.asarray(self.x),
            converged=jnp.asarray(self.converged),
            _maxcor=jnp.asarray(self._maxcor),
            _line_search_max_steps=jnp.asarray(self._line_search_max_steps),
            _ftol=jnp.asarray(self._ftol),
            _gtol=jnp.asarray(self._gtol),
            _wa=jnp.asarray(self._wa),
            _iwa=jnp.asarray(self._iwa),
            _task=_array_from_s60_str(self._task),
            _csave=_array_from_s60_str(self._csave),
            _lsave=jnp.asarray(self._lsave),
            _isave=jnp.asarray(self._isave),
            _dsave=jnp.asarray(self._dsave),
            _lower_bound=jnp.asarray(self._lower_bound),
            _upper_bound=jnp.asarray(self._upper_bound),
            _bound_type=jnp.asarray(self._bound_type),
        )

    @classmethod
    def from_jax(cls, state_dict: Dict[str, jnp.ndarray]) -> "ScipyLbfgsbState":
        """Converts a dictionary of jax arrays to a `ScipyLbfgsbState`."""
        return ScipyLbfgsbState(
            x=onp.array(state_dict["x"], dtype=onp.float64),
            converged=onp.array(state_dict["converged"], dtype=bool),
            _maxcor=int(state_dict["_maxcor"]),
            _line_search_max_steps=int(state_dict["_line_search_max_steps"]),
            _ftol=onp.array(state_dict["_ftol"], dtype=onp.float64),
            _gtol=onp.array(state_dict["_gtol"], dtype=onp.float64),
            _wa=onp.array(state_dict["_wa"], onp.float64),
            _iwa=onp.array(state_dict["_iwa"], dtype=FORTRAN_INT),
            _task=_s60_str_from_array(state_dict["_task"]),
            _csave=_s60_str_from_array(state_dict["_csave"]),
            _lsave=onp.array(state_dict["_lsave"], dtype=FORTRAN_INT),
            _isave=onp.array(state_dict["_isave"], dtype=FORTRAN_INT),
            _dsave=onp.array(state_dict["_dsave"], dtype=onp.float64),
            _lower_bound=onp.array(state_dict["_lower_bound"], dtype=onp.float64),
            _upper_bound=onp.array(state_dict["_upper_bound"], dtype=onp.float64),
            _bound_type=onp.array(state_dict["_bound_type"], dtype=int),
        )

    @classmethod
    def init(
        cls,
        x0: NDArray,
        lower_bound: ElementwiseBound,
        upper_bound: ElementwiseBound,
        maxcor: int,
        line_search_max_steps: int,
        ftol: float,
        gtol: float,
    ) -> "ScipyLbfgsbState":
        """Initializes the `ScipyLbfgsbState` for `x0`.

        Args:
            x0: Array giving the initial solution vector.
            lower_bound: Array giving the elementwise optional lower bound.
            upper_bound: Array giving the elementwise optional upper bound.
            maxcor: The maximum number of variable metric corrections used to define
                the limited memory matrix, in the L-BFGS-B scheme.
            line_search_max_steps: The maximum number of steps in the line search.
            ftol: Tolerance for stopping criteria based on function values. See scipy
                documentation for details.
            gtol: Tolerance for stopping criteria based on gradient.

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
        wa_size = _wa_size(n=n, maxcor=maxcor)
        state = ScipyLbfgsbState(
            x=onp.array(x0, onp.float64),
            converged=onp.asarray(False),
            _maxcor=maxcor,
            _line_search_max_steps=line_search_max_steps,
            _ftol=onp.asarray(ftol, onp.float64),
            _gtol=onp.asarray(gtol, onp.float64),
            _wa=onp.zeros(wa_size, onp.float64),
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
        """Performs an in-place update of the `ScipyLbfgsbState` if not converged.

        Args:
            grad: The function gradient for the current `x`.
            value: The scalar function value for the current `x`.
        """
        if self.converged:
            return
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
                factr=self._ftol / onp.finfo(float).eps,
                pgtol=self._gtol,
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
            if task_str.startswith(TASK_CONVERGED):
                self.converged = onp.asarray(True)
            if task_str.startswith(TASK_FG):
                break


def _wa_size(n: int, maxcor: int) -> int:
    """Return the size of the `wa` attribute of lbfgsb state."""
    return 2 * maxcor * n + 5 * n + 11 * maxcor**2 + 8 * maxcor


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


def _array_from_s60_str(s60_str: NDArray) -> jnp.ndarray:
    """Return a jax array for a numpy s60 string."""
    assert s60_str.shape == (1,)
    chars = [int(o) for o in s60_str[0]]
    chars.extend([32] * (59 - len(chars)))
    return jnp.asarray(chars, dtype=int)


def _s60_str_from_array(array: jnp.ndarray) -> NDArray:
    """Return a numpy s60 string for a jax array."""
    return onp.asarray(
        [b"".join(int(i).to_bytes(length=1, byteorder="big") for i in array)],
        dtype="S60",
    )
