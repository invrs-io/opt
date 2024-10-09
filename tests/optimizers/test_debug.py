"""Defines tests for the `lbfgsb.lbfgsb` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import scipy.optimize as spo
from jax import flatten_util
from parameterized import parameterized
from totypes import types

from invrs_opt.parameterization import filter_project
from invrs_opt.optimizers import lbfgsb

jax.config.update("jax_enable_x64", True)

def optimization_with_vmap(steps):
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
    opt = lbfgsb.density_lbfgsb(beta=2, maxcor=20)

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

    for i in range(steps):
        state, value = step_fn(state)

    # Test one-at-a-time optimization.
    for k in keys:
        params = initial_params_fn(k)
        state = opt.init(params)
        for _ in range(steps):
            params = opt.params(state)
            value, grad = jax.jit(jax.value_and_grad(loss_fn))(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)


class VmapTest(unittest.TestCase):
    def test_optimization_with_vmap(self):
        optimization_with_vmap(steps=10)

    def test_optimization_with_vmap_fewer_steps(self):
        optimization_with_vmap(steps=2)


# class DensityLbfgsbBoundsTest(unittest.TestCase):
#     @parameterized.expand([[-1, 1, 1], [-1, 1, -1], [0, 1, 1], [0, 1, -1]])
#     def test_respects_bounds(self, lower_bound, upper_bound, sign):
#         def loss_fn(density):
#             return sign * jnp.sum(density.array)

#         params = types.Density2DArray(
#             array=jnp.ones((5, 5)) * (lower_bound + upper_bound) / 2,
#             lower_bound=lower_bound,
#             upper_bound=upper_bound,
#         )
#         opt = lbfgsb.density_lbfgsb(beta=2.0)
#         state = opt.init(params)
#         for _ in range(10):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)

#         params = opt.params(state)
#         expected = upper_bound if sign < 0 else lower_bound
#         onp.testing.assert_allclose(params.array, expected)


# class DensityLbfgsbInitializeTest(unittest.TestCase):
#     @parameterized.expand(
#         [
#             [-1, 1, -0.95],
#             [-1, 1, -0.50],
#             [-1, 1, 0.00],
#             [-1, 1, 0.50],
#             [-1, 1, 0.95],
#             [0, 1, 0.05],
#             [0, 1, 0.25],
#             [0, 1, 0.00],
#             [0, 1, 0.25],
#             [0, 1, 0.95],
#         ]
#     )
#     def test_initial_params_match_expected(self, lb, ub, value):
#         density = types.Density2DArray(
#             array=jnp.full((10, 10), value),
#             lower_bound=lb,
#             upper_bound=ub,
#         )
#         opt = lbfgsb.density_lbfgsb(beta=4)
#         state = opt.init(density)
#         params = opt.params(state)
#         onp.testing.assert_allclose(density.array, params.array, atol=1e-2)

#     def test_initial_params_out_of_bounds(self):
#         density = types.Density2DArray(
#             array=jnp.full((10, 10), 10),
#             lower_bound=-1,
#             upper_bound=1,
#         )
#         opt = lbfgsb.density_lbfgsb(beta=4)
#         state = opt.init(density)
#         params = opt.params(state)
#         onp.testing.assert_allclose(params.array, onp.ones_like(params.array))


# class LbfgsbBoundsTest(unittest.TestCase):
#     @parameterized.expand([[-1, 1, 1], [-1, 1, -1], [0, 1, 1], [0, 1, -1]])
#     def test_respects_bounds(self, lower_bound, upper_bound, sign):
#         def loss_fn(density):
#             return sign * jnp.sum(density.array)

#         params = types.Density2DArray(
#             array=jnp.ones((5, 5)) * (lower_bound + upper_bound) / 2,
#             lower_bound=lower_bound,
#             upper_bound=upper_bound,
#         )
#         opt = lbfgsb.lbfgsb()
#         state = opt.init(params)
#         for _ in range(10):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)

#         params = opt.params(state)
#         expected = upper_bound if sign < 0 else lower_bound
#         onp.testing.assert_allclose(params.array, expected)


# class LbfgsbInputValidationTest(unittest.TestCase):
#     @parameterized.expand([[0], [-1], [500], ["abc"]])
#     def test_maxcor_validation(self, invalid_maxcor):
#         with self.assertRaisesRegex(ValueError, "`maxcor` must be greater than 0"):
#             lbfgsb.lbfgsb(maxcor=invalid_maxcor, line_search_max_steps=100)

#     @parameterized.expand([[0], [-1], ["abc"]])
#     def test_line_search_max_steps_validation(self, invalid_line_search_max_steps):
#         with self.assertRaisesRegex(ValueError, "`line_search_max_steps` must be "):
#             lbfgsb.lbfgsb(
#                 maxcor=20, line_search_max_steps=invalid_line_search_max_steps
#             )


# class LbfgsbInitializeTest(unittest.TestCase):
#     @parameterized.expand(
#         [
#             [2.0],
#             [jnp.ones((3,))],
#             [
#                 types.BoundedArray(
#                     array=jnp.ones((3,)),
#                     lower_bound=0.0,
#                     upper_bound=1.0,
#                 )
#             ],
#             [
#                 types.BoundedArray(
#                     array=jnp.ones((3,)),
#                     lower_bound=None,
#                     upper_bound=1.0,
#                 )
#             ],
#             [
#                 types.BoundedArray(
#                     array=jnp.ones((3,)),
#                     lower_bound=None,
#                     upper_bound=None,
#                 )
#             ],
#             [
#                 types.BoundedArray(
#                     array=jnp.ones((3,)),
#                     lower_bound=None,
#                     upper_bound=None,
#                 )
#             ],
#             [
#                 types.BoundedArray(
#                     array=jnp.ones((3,)),
#                     lower_bound=jnp.zeros((3,)),
#                     upper_bound=jnp.ones((3,)),
#                 )
#             ],
#             [
#                 types.Density2DArray(
#                     array=jnp.ones((3, 3)),
#                     lower_bound=0.0,
#                     upper_bound=1.0,
#                     fixed_solid=None,
#                     fixed_void=None,
#                     minimum_width=1,
#                     minimum_spacing=1,
#                 )
#             ],
#             [
#                 {
#                     "a": types.Density2DArray(
#                         array=jnp.ones((3, 3)),
#                         lower_bound=0.0,
#                         upper_bound=1.0,
#                         fixed_solid=None,
#                         fixed_void=None,
#                         minimum_width=1,
#                         minimum_spacing=1,
#                     ),
#                     "b": types.BoundedArray(
#                         array=jnp.ones((3,)),
#                         lower_bound=jnp.zeros((3,)),
#                         upper_bound=jnp.ones((3,)),
#                     ),
#                 }
#             ],
#             [
#                 {
#                     "a": types.Density2DArray(
#                         array=jnp.ones((3, 3)),
#                         lower_bound=0.0,
#                         upper_bound=1.0,
#                         fixed_solid=None,
#                         fixed_void=None,
#                         minimum_width=1,
#                         minimum_spacing=1,
#                     ),
#                     "b": None,
#                 }
#             ],
#         ]
#     )
#     def test_initialize(self, params):
#         opt = lbfgsb.density_lbfgsb(maxcor=20, line_search_max_steps=100, beta=2.0)
#         state = opt.init(params)
#         params = opt.params(state)
#         dummy_grad = jax.tree_util.tree_map(jnp.zeros_like, params)
#         state = opt.update(value=0.0, params=params, grad=dummy_grad, state=state)


# class LbfgsbTest(unittest.TestCase):
#     def test_trajectory_matches_scipy_bounded_array(self):
#         # Define a random quadratic objective function.
#         onp.random.seed(0)
#         xc = onp.random.randn(20)
#         scale = onp.random.rand(20)
#         x0 = onp.random.randn(20)

#         lb_ub = onp.random.randn(2, 20)
#         lb, ub = onp.sort(lb_ub, axis=0)
#         assert onp.all(lb < ub)

#         values = []

#         def value_fn(x):
#             nonlocal values
#             value = onp.sum(scale * (x - xc) ** 2)
#             values.append(value)
#             return value

#         def grad_fn(x):
#             return 2 * scale * (x - xc)

#         # Carry out the optimization directly using scipy's L-BFGS-B implementation.
#         spo.minimize(
#             fun=value_fn,
#             x0=x0,
#             jac=grad_fn,
#             method="L-BFGS-B",
#             bounds=list(zip(lb, ub)),
#         )
#         scipy_values = onp.asarray(values)

#         # Carry out the optimization using our wrapper. Reformulate the problem so
#         # that our parameters are a pytree rather than a vector.
#         xc = {"a": xc[:5], "b": xc[5:10], "c": xc[10:].reshape((2, 5))}
#         scale = {"a": scale[:5], "b": scale[5:10], "c": scale[10:].reshape((2, 5))}
#         x0 = {
#             "a": types.BoundedArray(
#                 array=x0[:5], lower_bound=lb[:5], upper_bound=ub[:5]
#             ),
#             "b": types.BoundedArray(
#                 array=x0[5:10], lower_bound=lb[5:10], upper_bound=ub[5:10]
#             ),
#             "c": types.BoundedArray(
#                 array=x0[10:].reshape((2, 5)),
#                 lower_bound=lb[10:].reshape((2, 5)),
#                 upper_bound=ub[10:].reshape((2, 5)),
#             ),
#         }

#         def loss_fn(x):
#             x_leaves = jax.tree_util.tree_leaves(x)
#             xc_leaves = jax.tree_util.tree_leaves(xc)
#             scale_leaves = jax.tree_util.tree_leaves(scale)
#             return jnp.sum(
#                 jnp.asarray(
#                     [
#                         jnp.sum((x - c) ** 2 * s)
#                         for x, c, s in zip(x_leaves, xc_leaves, scale_leaves)
#                     ]
#                 )
#             )

#         opt = lbfgsb.lbfgsb(maxcor=20, line_search_max_steps=100)
#         state = opt.init(x0)
#         num_steps = len(scipy_values)
#         wrapper_values = []
#         for _ in range(num_steps):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)
#             wrapper_values.append(value)
#         wrapper_values = onp.asarray(wrapper_values)

#         # Compare the first few steps for the two schemes. We expect some slight
#         # numerical differences, since in once case we are using float64 and in
#         # the other we are using float32.
#         onp.testing.assert_allclose(scipy_values[:10], wrapper_values[:10], rtol=1e-6)

#     def test_trajectory_matches_scipy_density_2d(self):
#         # Define a random quadratic objective function.
#         onp.random.seed(0)
#         xc = onp.random.randn(20)
#         scale = onp.random.rand(20)
#         x0 = onp.random.randn(20)

#         lb = -1.0
#         ub = 1.0

#         values = []

#         def value_fn(x):
#             nonlocal values
#             value = onp.sum(scale * (x - xc) ** 2)
#             values.append(value)
#             return value

#         def grad_fn(x):
#             return 2 * scale * (x - xc)

#         # Carry out the optimization directly using scipy's L-BFGS-B implementation.
#         spo.minimize(
#             fun=value_fn,
#             x0=x0,
#             jac=grad_fn,
#             method="L-BFGS-B",
#             bounds=list(zip(onp.full_like(x0, lb), onp.full_like(x0, ub))),
#         )
#         scipy_values = onp.asarray(values)

#         # Carry out the optimization using our wrapper. Reformulate the problem so
#         # that our parameters are a pytree rather than a vector.
#         xc = xc.reshape((4, 5))
#         scale = scale.reshape((4, 5))
#         x0 = types.Density2DArray(
#             array=x0.reshape((4, 5)),
#             lower_bound=lb,
#             upper_bound=ub,
#             fixed_solid=None,
#             fixed_void=None,
#             minimum_width=1,
#             minimum_spacing=1,
#         )

#         def loss_fn(x):
#             x_leaves = jax.tree_util.tree_leaves(x)
#             xc_leaves = jax.tree_util.tree_leaves(xc)
#             scale_leaves = jax.tree_util.tree_leaves(scale)
#             return jnp.sum(
#                 jnp.asarray(
#                     [
#                         jnp.sum((x - c) ** 2 * s)
#                         for x, c, s in zip(x_leaves, xc_leaves, scale_leaves)
#                     ]
#                 )
#             )

#         opt = lbfgsb.lbfgsb(maxcor=20, line_search_max_steps=100)
#         state = opt.init(x0)
#         num_steps = len(scipy_values)
#         wrapper_values = []
#         for _ in range(num_steps):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)
#             wrapper_values.append(value)
#         wrapper_values = onp.asarray(wrapper_values)

#         # Compare the first few steps for the two schemes. We expect some slight
#         # numerical differences, since in once case we are using float64 and in
#         # the other we are using float32.
#         onp.testing.assert_allclose(scipy_values[:10], wrapper_values[:10], rtol=1e-6)

#     def test_density_lbfgsb_reaches_bounds(self):
#         def loss_fn(density):
#             return jnp.sum(jnp.abs(density.array - 1) ** 2)

#         opt = lbfgsb.density_lbfgsb(maxcor=20, line_search_max_steps=100, beta=2.0)

#         density = types.Density2DArray(
#             array=jnp.zeros((3, 3)),
#             lower_bound=0.0,
#             upper_bound=1.0,
#             fixed_solid=None,
#             fixed_void=None,
#             minimum_width=1,
#             minimum_spacing=1,
#         )
#         state = opt.init(density)
#         for _ in range(20):
#             density = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(density)
#             state = opt.update(value=value, grad=grad, params=density, state=state)

#         onp.testing.assert_allclose(density.array, 1.0)

#     def test_optimization_with_vmap(self):
#         def initial_params_fn(key):
#             ka, kb = jax.random.split(key)
#             return {
#                 "a": jax.random.normal(ka, (10,)),
#                 "b": jax.random.normal(kb, (10,)),
#                 "c": types.Density2DArray(array=jnp.ones((3, 3))),
#             }

#         def loss_fn(params):
#             flat, _ = flatten_util.ravel_pytree(params)
#             return jnp.sum(jnp.abs(flat**2))

#         keys = jax.random.split(jax.random.PRNGKey(0))
#         opt = lbfgsb.density_lbfgsb(beta=2, maxcor=20)

#         # Test batch optimization
#         params = jax.vmap(initial_params_fn)(keys)
#         state = jax.vmap(opt.init)(params)

#         @jax.jit
#         @jax.vmap
#         def step_fn(state):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)
#             return state, value

#         batch_values = []
#         for i in range(10):
#             state, value = step_fn(state)
#             batch_values.append(value)

#         # Test one-at-a-time optimization.
#         no_batch_values = []
#         for k in keys:
#             no_batch_values.append([])
#             params = initial_params_fn(k)
#             state = opt.init(params)
#             for _ in range(10):
#                 params = opt.params(state)
#                 value, grad = jax.jit(jax.value_and_grad(loss_fn))(params)
#                 state = opt.update(grad=grad, value=value, params=params, state=state)
#                 no_batch_values[-1].append(value)

#         onp.testing.assert_allclose(
#             batch_values, onp.transpose(no_batch_values, (1, 0)), atol=1e-4
#         )

#     def test_converged(self):
#         def loss_fn(x):
#             return jnp.sum(x**2)

#         x = jnp.ones((5,))

#         opt = lbfgsb.lbfgsb()
#         state = opt.init(x)

#         for i in range(100):
#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             state = opt.update(grad=grad, value=value, params=params, state=state)
#             if lbfgsb.is_converged(state):
#                 break

#         # The optimization should converge in fewer than 20 steps.
#         self.assertLess(i, 20)


# class BoundsForParamsTest(unittest.TestCase):
#     def test_none_bounds(self):
#         params = {
#             "a": jnp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": jnp.asarray([1.0, 2.0]), "d": jnp.asarray([5.0, 1.0])},
#         }
#         bounds = lbfgsb._bound_for_params(None, params)
#         expected_bounds = [None] * 7
#         onp.testing.assert_array_equal(bounds, expected_bounds)

#     def test_scalar_bounds(self):
#         params = {
#             "a": jnp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": jnp.asarray([1.0, 2.0]), "d": jnp.asarray([5.0, 1.0])},
#         }
#         bounds = lbfgsb._bound_for_params(-1, params)
#         expected_bounds = [-1] * 7
#         onp.testing.assert_array_equal(bounds, expected_bounds)

#     def test_scalar_bounds_per_array(self):
#         params = {
#             "a": jnp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": jnp.asarray([1.0, 2.0]), "d": jnp.asarray([5.0, 1.0])},
#         }
#         bounds_tree = {"a": None, "b": {"c": -2, "d": 1}}
#         bounds = lbfgsb._bound_for_params(bounds_tree, params)
#         expected_bounds = [None, None, None, -2, -2, 1, 1]
#         onp.testing.assert_array_equal(bounds, expected_bounds)

#     def test_array_bounds_per_array(self):
#         params = {
#             "a": jnp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": jnp.asarray([1.0, 2.0]), "d": jnp.asarray([5.0, 1.0])},
#         }
#         bounds_tree = {"a": jnp.asarray([-1, -2, -3]), "b": {"c": -2, "d": None}}
#         bounds = lbfgsb._bound_for_params(bounds_tree, params)
#         expected_bounds = [-1, -2, -3, -2, -2, None, None]
#         onp.testing.assert_array_equal(bounds, expected_bounds)


# class ConverterTest(unittest.TestCase):
#     def test_converter_jax_pytree(self):
#         params = {
#             "a": jnp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": jnp.asarray([1.0, 2.0]), "d": jnp.asarray([5.0, 1.0])},
#         }
#         x = lbfgsb._to_numpy(params)
#         self.assertEqual(x.dtype, onp.float64)
#         self.assertSequenceEqual(x.shape, (7,))
#         restored_params = lbfgsb._to_pytree(x, params)
#         self.assertEqual(
#             jax.tree_util.tree_structure(params),
#             jax.tree_util.tree_structure(restored_params),
#         )
#         for a, b in zip(
#             jax.tree_util.tree_leaves(params),
#             jax.tree_util.tree_leaves(restored_params),
#         ):
#             self.assertTrue(isinstance(a, jnp.ndarray))
#             self.assertTrue(isinstance(b, jnp.ndarray))
#             onp.testing.assert_array_equal(a, b)

#     def test_converter_numpy_pytree(self):
#         params = {
#             "a": onp.asarray([1.0, 2.0, 3.0]),
#             "b": {"c": onp.asarray([1.0, 2.0]), "d": onp.asarray([5.0, 1.0])},
#         }
#         x = lbfgsb._to_numpy(params)
#         self.assertEqual(x.dtype, onp.float64)
#         self.assertSequenceEqual(x.shape, (7,))
#         restored_params = lbfgsb._to_pytree(x, params)
#         self.assertEqual(
#             jax.tree_util.tree_structure(params),
#             jax.tree_util.tree_structure(restored_params),
#         )
#         for a, b in zip(
#             jax.tree_util.tree_leaves(params),
#             jax.tree_util.tree_leaves(restored_params),
#         ):
#             self.assertTrue(isinstance(a, onp.ndarray))
#             self.assertTrue(isinstance(b, jnp.ndarray))
#             onp.testing.assert_array_equal(a, b)


# class ScipyLbfgsStateTest(unittest.TestCase):
#     def test_x0_shape_validation(self):
#         with self.assertRaisesRegex(ValueError, "`x0` must be rank-1 but got"):
#             lbfgsb.ScipyLbfgsbState.init(
#                 x0=onp.ones((2, 2)),
#                 lower_bound=onp.zeros((2, 2)),
#                 upper_bound=onp.ones((2, 2)),
#                 maxcor=20,
#                 line_search_max_steps=100,
#                 ftol=0,
#                 gtol=0,
#             )

#     def test_lower_bound_shape_validation(self):
#         with self.assertRaisesRegex(ValueError, "`x0`, `lower_bound`, and "):
#             lbfgsb.ScipyLbfgsbState.init(
#                 x0=onp.ones((4,)),
#                 lower_bound=onp.zeros((3,)),
#                 upper_bound=onp.ones((4,)),
#                 maxcor=20,
#                 line_search_max_steps=100,
#                 ftol=0,
#                 gtol=0,
#             )

#     def test_upper_bound_shape_validation(self):
#         with self.assertRaisesRegex(ValueError, "`x0`, `lower_bound`, and "):
#             lbfgsb.ScipyLbfgsbState.init(
#                 x0=onp.ones((4,)),
#                 lower_bound=onp.zeros((4,)),
#                 upper_bound=onp.ones((3,)),
#                 maxcor=20,
#                 line_search_max_steps=100,
#                 ftol=0,
#                 gtol=0,
#             )

#     def test_maxcor_positive(self):
#         with self.assertRaisesRegex(ValueError, "`maxcor` must be positive but"):
#             lbfgsb.ScipyLbfgsbState.init(
#                 x0=onp.ones((4,)),
#                 lower_bound=onp.zeros((4,)),
#                 upper_bound=onp.ones((4,)),
#                 maxcor=0,
#                 line_search_max_steps=100,
#                 ftol=0,
#                 gtol=0,
#             )

#     def test_unbounded_trajectory_matches_scipy(self):
#         onp.random.seed(0)
#         xc = onp.random.randn(10)
#         scale = onp.random.rand(10)
#         x0 = onp.random.randn(10)

#         values = []

#         def value_fn(x):
#             nonlocal values
#             value = onp.sum(scale * (x - xc) ** 2)
#             values.append(value)
#             return value

#         def grad_fn(x):
#             return 2 * scale * x

#         # Carry out the optimization directly using scipy's L-BFGS-B implementation.
#         _ = spo.minimize(
#             fun=value_fn, x0=x0, jac=grad_fn, method="L-BFGS-B", bounds=None
#         )
#         scipy_values = onp.asarray(values)

#         # Carry out the optimization using our wrapper.
#         state = lbfgsb.ScipyLbfgsbState.init(
#             x0=x0,
#             lower_bound=[None] * 10,
#             upper_bound=[None] * 10,
#             maxcor=20,
#             line_search_max_steps=100,
#             ftol=0,
#             gtol=0,
#         )
#         num_steps = len(scipy_values)
#         wrapper_values = []
#         for _ in range(num_steps):
#             value = value_fn(state.x)
#             grad = grad_fn(state.x)
#             state.update(grad, value)
#             wrapper_values.append(value)
#         wrapper_values = onp.asarray(wrapper_values)

#         # Compare the first few steps for the two schemes.
#         onp.testing.assert_allclose(scipy_values[:10], wrapper_values[:10])

#     def test_bounded_trajectory_matches_scipy(self):
#         onp.random.seed(42)
#         xc = onp.random.randn(10)
#         scale = onp.random.rand(10)
#         x0 = onp.random.randn(10)

#         lb_ub = onp.random.randn(2, 10)
#         lb, ub = onp.sort(lb_ub, axis=0)

#         values = []

#         def value_fn(x):
#             nonlocal values
#             value = onp.sum(scale * (x - xc) ** 2)
#             values.append(value)
#             return value

#         def grad_fn(x):
#             return 2 * scale * x

#         # Carry out the optimization directly using scipy's L-BFGS-B implementation.
#         _ = spo.minimize(
#             fun=value_fn,
#             x0=x0,
#             jac=grad_fn,
#             method="L-BFGS-B",
#             bounds=list(zip(lb, ub)),
#         )
#         scipy_values = onp.asarray(values)

#         # Carry out the optimization using our wrapper.
#         state = lbfgsb.ScipyLbfgsbState.init(
#             x0=x0,
#             lower_bound=lb,
#             upper_bound=ub,
#             maxcor=20,
#             line_search_max_steps=100,
#             ftol=0,
#             gtol=0,
#         )
#         num_steps = len(scipy_values)
#         wrapper_values = []
#         for _ in range(num_steps):
#             value = value_fn(state.x)
#             grad = grad_fn(state.x)
#             state.update(grad, value)
#             wrapper_values.append(value)
#         wrapper_values = onp.asarray(wrapper_values)

#         # Compare the first few steps for the two schemes.
#         onp.testing.assert_allclose(scipy_values[:10], wrapper_values[:10])


# class StepVariableParameterizationTest(unittest.TestCase):
#     def test_variable_parameterization(self):
#         # Create a custom parameterization whose update method increments `beta` by 1
#         # at each step.
#         p = filter_project.filter_project(beta=1)

#         def update_fn(params, step):
#             del step
#             params.metadata.beta += 1
#             return params

#         p.update = update_fn

#         opt = lbfgsb.parameterized_lbfgsb(
#             density_parameterization=p,
#             penalty=1.0,
#         )

#         target = jnp.asarray([[0, 1], [1, 0]], dtype=float)
#         target = jnp.kron(target, jnp.ones((10, 10)))

#         density = types.Density2DArray(
#             array=jnp.full(target.shape, 0.5),
#             lower_bound=0,
#             upper_bound=1,
#             minimum_width=4,
#             minimum_spacing=4,
#         )

#         state = opt.init(density)

#         def step_fn(state):
#             def loss_fn(density):
#                 return jnp.sum((density.array - target) ** 2)

#             params = opt.params(state)
#             value, grad = jax.value_and_grad(loss_fn)(params)
#             return opt.update(grad=grad, value=value, params=params, state=state)

#         for _ in range(10):
#             state = step_fn(state)

#         # Check that beta has actually been incremented.
#         self.assertEqual(state[2].metadata.beta, 11)
