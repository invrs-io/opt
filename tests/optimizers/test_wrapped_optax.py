"""Defines tests for the `wrapped_optax.wrapped_optax` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import copy
import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import optax
from jax import flatten_util, tree_util
from parameterized import parameterized
from totypes import types

from invrs_opt.parameterization import filter_project, transforms
from invrs_opt.optimizers import wrapped_optax


class DensityWrappedOptaxBoundsTest(unittest.TestCase):
    @parameterized.expand([[-1, 1, 1], [-1, 1, -1], [0, 1, 1], [0, 1, -1]])
    def test_respects_bounds(self, lower_bound, upper_bound, sign):
        def loss_fn(density):
            return sign * jnp.sum(density.array)

        params = types.Density2DArray(
            array=jnp.ones((5, 5)) * (lower_bound + upper_bound) / 2,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        opt = wrapped_optax.density_wrapped_optax(opt=optax.adam(0.1), beta=2)
        state = opt.init(params)
        for _ in range(20):
            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)

        params = opt.params(state)
        expected = upper_bound if sign < 0 else lower_bound
        onp.testing.assert_allclose(params.array, expected, atol=1e-5)


class DensityWrappedOptaxInitializeTest(unittest.TestCase):
    @parameterized.expand(
        [
            [-1, 1, -0.95],
            [-1, 1, -0.50],
            [-1, 1, 0.00],
            [-1, 1, 0.50],
            [-1, 1, 0.95],
            [0, 1, 0.05],
            [0, 1, 0.25],
            [0, 1, 0.00],
            [0, 1, 0.25],
            [0, 1, 0.95],
        ]
    )
    def test_initial_params_match_expected(self, lb, ub, value):
        density = types.Density2DArray(
            array=jnp.full((10, 10), value),
            lower_bound=lb,
            upper_bound=ub,
        )
        opt = wrapped_optax.density_wrapped_optax(optax.adam(0.01), beta=4)
        state = opt.init(density)
        params = opt.params(state)
        onp.testing.assert_allclose(density.array, params.array, atol=1e-2)

    def test_initial_params_out_of_bounds(self):
        density = types.Density2DArray(
            array=jnp.full((10, 10), 10),
            lower_bound=-1,
            upper_bound=1,
        )
        opt = wrapped_optax.density_wrapped_optax(optax.adam(0.01), beta=4)
        state = opt.init(density)
        params = opt.params(state)
        onp.testing.assert_allclose(params.array, onp.ones_like(params.array))


class WrappedOptaxBoundsTest(unittest.TestCase):
    @parameterized.expand([[-1, 1, 1], [-1, 1, -1], [0, 1, 1], [0, 1, -1]])
    def test_respects_bounds(self, lower_bound, upper_bound, sign):
        def loss_fn(density):
            return sign * jnp.sum(density.array)

        params = types.Density2DArray(
            array=jnp.ones((5, 5)) * (lower_bound + upper_bound) / 2,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        opt = wrapped_optax.wrapped_optax(optax.adam(0.1))
        state = opt.init(params)
        for _ in range(10):
            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)

        params = opt.params(state)
        expected = upper_bound if sign < 0 else lower_bound
        onp.testing.assert_allclose(params.array, expected, atol=1e-5)


class WrappedOptaxTest(unittest.TestCase):
    def test_trajectory_matches_optax_bounded_array(self):
        initial_params = {
            "a": jnp.asarray([1.0, 2.0]),
            "b": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=2.0, upper_bound=None
            ),
            "c": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=2.0, upper_bound=5.0
            ),
            "d": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=None, upper_bound=5.0
            ),
        }

        def loss_fn(params):
            x, _ = flatten_util.ravel_pytree(params)
            return jnp.sum(x**2)

        # Carry out optimization directly
        opt = optax.adam(1e-2)
        params = wrapped_optax._clip(initial_params)
        state = opt.init(params)

        expected_values = []
        for _ in range(10):
            value, grad = jax.value_and_grad(loss_fn)(params)
            updates, state = opt.update(grad, state, params=params)
            params = wrapped_optax._clip(optax.apply_updates(params, updates=updates))
            expected_values.append(value)

        # Carry out optimization using the wrapped optimizer.
        wrapped_opt = wrapped_optax.wrapped_optax(opt)
        state = wrapped_opt.init(initial_params)

        values = []
        for _ in range(10):
            params = wrapped_opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = wrapped_opt.update(
                grad=grad, value=value, state=state, params=params
            )
            values.append(value)

        onp.testing.assert_array_equal(values, expected_values)

    def test_trajectory_matches_optax_density_2d(self):
        initial_params = {
            "a": jnp.asarray([1.0, 2.0]),
            "b": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=2.0, upper_bound=None
            ),
            "c": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=2.0, upper_bound=5.0
            ),
            "d": types.BoundedArray(
                jnp.asarray([3.0, 4.0]), lower_bound=None, upper_bound=5.0
            ),
            "density": types.Density2DArray(jnp.arange(20, dtype=float).reshape(4, 5)),
        }
        beta = 2.0

        def loss_fn(params):
            x, _ = flatten_util.ravel_pytree(params)
            return jnp.sum(x**2)

        def latent_loss_fn(params):
            params["density"] = transform_density(params["density"])
            return loss_fn(params)

        def transform_density(density):
            transformed = types.symmetrize_density(density)
            transformed = transforms.density_gaussian_filter_and_tanh(
                transformed, beta=beta
            )
            # Scale to ensure that full valid range of the density array is reachable.
            mid_value = (density.lower_bound + density.upper_bound) / 2
            transformed = tree_util.tree_map(
                lambda array: mid_value + (array - mid_value) / jnp.tanh(beta),
                transformed,
            )
            return transforms.apply_fixed_pixels(transformed)

        def initialize_latent_density(density) -> types.Density2DArray:
            array = transforms.normalized_array_from_density(density)
            array = jnp.clip(array, -1, 1)
            array *= jnp.tanh(beta)
            latent_array = jnp.arctanh(array) / beta
            latent_array = transforms.rescale_array_for_density(latent_array, density)
            return dataclasses.replace(density, array=latent_array)

        # Carry out optimization directly
        opt = optax.adam(1e-2)
        params = wrapped_optax._clip(initial_params)
        params["density"] = initialize_latent_density(params["density"])
        state = opt.init(params)

        expected_values = []
        for _ in range(10):
            value, grad = jax.value_and_grad(latent_loss_fn)(params)
            updates, state = opt.update(grad, state, params=params)
            params = wrapped_optax._clip(optax.apply_updates(params, updates=updates))
            expected_values.append(value)

        # Carry out optimization using the wrapped optimizer.
        wrapped_opt = wrapped_optax.density_wrapped_optax(opt, beta=beta)
        state = wrapped_opt.init(initial_params)

        values = []
        for _ in range(10):
            params = wrapped_opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = wrapped_opt.update(
                grad=grad, value=value, state=state, params=params
            )
            values.append(value)

        onp.testing.assert_array_equal(values, expected_values)

    @parameterized.expand(
        [
            [2.0],
            [jnp.ones((3,))],
            [
                types.BoundedArray(
                    array=jnp.ones((3,)),
                    lower_bound=0.0,
                    upper_bound=1.0,
                )
            ],
            [
                types.BoundedArray(
                    array=jnp.ones((3,)),
                    lower_bound=None,
                    upper_bound=1.0,
                )
            ],
            [
                types.BoundedArray(
                    array=jnp.ones((3,)),
                    lower_bound=None,
                    upper_bound=None,
                )
            ],
            [
                types.BoundedArray(
                    array=jnp.ones((3,)),
                    lower_bound=None,
                    upper_bound=None,
                )
            ],
            [
                types.BoundedArray(
                    array=jnp.ones((3,)),
                    lower_bound=jnp.zeros((3,)),
                    upper_bound=jnp.ones((3,)),
                )
            ],
            [
                types.Density2DArray(
                    array=jnp.ones((3, 3)),
                    lower_bound=0.0,
                    upper_bound=1.0,
                    fixed_solid=None,
                    fixed_void=None,
                    minimum_width=1,
                    minimum_spacing=1,
                )
            ],
            [
                {
                    "a": types.Density2DArray(
                        array=jnp.ones((3, 3)),
                        lower_bound=0.0,
                        upper_bound=1.0,
                        fixed_solid=None,
                        fixed_void=None,
                        minimum_width=1,
                        minimum_spacing=1,
                    ),
                    "b": types.BoundedArray(
                        array=jnp.ones((3,)),
                        lower_bound=jnp.zeros((3,)),
                        upper_bound=jnp.ones((3,)),
                    ),
                }
            ],
            [
                {
                    "a": types.Density2DArray(
                        array=jnp.ones((3, 3)),
                        lower_bound=0.0,
                        upper_bound=1.0,
                        fixed_solid=None,
                        fixed_void=None,
                        minimum_width=1,
                        minimum_spacing=1,
                    ),
                    "b": None,
                }
            ],
        ]
    )
    def test_initialize(self, params):
        opt = wrapped_optax.density_wrapped_optax(optax.adam(0.1), beta=2.0)
        state = opt.init(params)
        params = opt.params(state)
        dummy_grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        state = opt.update(value=0.0, params=params, grad=dummy_grad, state=state)

    def test_density_wrapped_optax_reaches_bounds(self):
        def loss_fn(density):
            return jnp.sum(jnp.abs(density.array - 1) ** 2)

        opt = wrapped_optax.density_wrapped_optax(optax.adam(0.1), beta=2.0)

        density = types.Density2DArray(
            array=jnp.zeros((3, 3)),
            lower_bound=0.0,
            upper_bound=1.0,
            fixed_solid=None,
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=1,
        )
        state = opt.init(density)
        for _ in range(20):
            density = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(density)
            state = opt.update(value=value, grad=grad, params=density, state=state)

        onp.testing.assert_allclose(density.array, 1.0)

    def test_optimization_with_vmap(self):
        def initial_params_fn(key):
            ka, kb = jax.random.split(key)
            return {
                "a": jax.random.normal(ka, (10,)),
                "b": jax.random.normal(kb, (10,)),
                "c": types.Density2DArray(array=jnp.ones((3, 3))),
            }

        def loss_fn(params):
            flat, _ = flatten_util.ravel_pytree(params)
            return jnp.sum(jnp.abs(flat**2))

        keys = jax.random.split(jax.random.PRNGKey(0))
        opt = wrapped_optax.density_wrapped_optax(optax.adam(0.1), beta=2.0)

        # Test batch optimization
        params = jax.vmap(initial_params_fn)(keys)
        state = jax.vmap(opt.init)(params)

        @jax.jit
        @jax.vmap
        def step_fn(state):
            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)
            return state, value

        batch_values = []
        for i in range(10):
            state, value = step_fn(state)
            batch_values.append(value)

        # Test one-at-a-time optimization.
        no_batch_values = []
        for k in keys:
            no_batch_values.append([])
            params = initial_params_fn(k)
            state = opt.init(params)
            for _ in range(10):
                params = opt.params(state)
                value, grad = jax.jit(jax.value_and_grad(loss_fn))(params)
                state = opt.update(grad=grad, value=value, params=params, state=state)
                no_batch_values[-1].append(value)

        onp.testing.assert_allclose(
            batch_values, onp.transpose(no_batch_values, (1, 0)), atol=1e-4
        )


class StepVariableParameterizationTest(unittest.TestCase):
    def test_variable_parameterization(self):
        # Create a custom parameterization whose update method increments `beta` by 1
        # at each step.
        p = filter_project.filter_project(beta=1)

        _original_update_fn = copy.deepcopy(p.update)

        def update_fn(step, params, value, updates):
            params = _original_update_fn(
                step=step, params=params, value=value, updates=updates
            )
            params.metadata.beta += 1
            return params

        p.update = update_fn

        opt = wrapped_optax.parameterized_wrapped_optax(
            opt=optax.adam(0.01),
            density_parameterization=p,
            penalty=1.0,
        )

        target = jnp.asarray([[0, 1], [1, 0]], dtype=float)
        target = jnp.kron(target, jnp.ones((10, 10)))

        density = types.Density2DArray(
            array=jnp.full(target.shape, 0.5),
            lower_bound=0,
            upper_bound=1,
            minimum_width=4,
            minimum_spacing=4,
        )

        state = opt.init(density)

        def step_fn(state):
            def loss_fn(density):
                return jnp.sum((density.array - target) ** 2)

            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            return opt.update(grad=grad, value=value, params=params, state=state)

        for _ in range(10):
            state = step_fn(state)

        # Check that beta has actually been incremented.
        self.assertEqual(state.latent_params.metadata.beta, 11)
