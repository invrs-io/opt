"""Defines tests across all optimization algorithms.

Copyright (c) 2023 The INVRS-IO authors.
"""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized
from totypes import json_utils, symmetry, types

import invrs_opt

jax.config.update("jax_enable_x64", True)


# Optimizers tested in this module.
OPTIMIZERS = [
    invrs_opt.lbfgsb(maxcor=20, line_search_max_steps=100),
    invrs_opt.density_lbfgsb(maxcor=20, line_search_max_steps=100, beta=2.0),
]

# Various parameter combinations tested in this module.
PARAMS_SCALAR = {"a": 1.0}
PARAMS_BASIC = (
    {
        "a": onp.asarray([1.0, 2.0, 3.0]),
        "b": (4.0, 5.0, 6.0),
    },
    onp.asarray([[7.0, 8.0]]),
)
PARAMS_WITH_BOUNDED_ARRAY_NUMPY = (
    {
        "a": onp.asarray([1.0, 2.0, 3.0]),
        "b": (4.0, 5.0, 6.0),
    },
    onp.asarray([[7.0, 8.0]]),
    {
        "scalar": types.BoundedArray(
            array=jnp.asarray(2.0),
            lower_bound=1.0,
            upper_bound=3.0,
        ),
        "vector_no_bounds": types.BoundedArray(
            array=onp.asarray([9.0, 10.0, 11.0]),
            lower_bound=None,
            upper_bound=None,
        ),
        "vector_scalar_lower_bound": types.BoundedArray(
            array=onp.asarray([12.0, 13.0, 14.0]),
            lower_bound=0.0,
            upper_bound=None,
        ),
        "vector_elementwise_lower_bound": types.BoundedArray(
            array=onp.asarray([15.0, 16.0, 17.0]),
            lower_bound=onp.asarray([0.0, 0.0, 0.0]),
            upper_bound=None,
        ),
        "vector_scalar_upper_bound": types.BoundedArray(
            array=onp.asarray([18.0, 19.0, 20.0]),
            lower_bound=None,
            upper_bound=100.0,
        ),
        "vector_elementwise_upper_bound": types.BoundedArray(
            array=onp.asarray([21.0, 22.0, 23.0]),
            lower_bound=None,
            upper_bound=onp.asarray([100.0, 100.0, 100.0]),
        ),
    },
)
PARAMS_WITH_BOUNDED_ARRAY_JAX = jax.tree_util.tree_map(
    jnp.asarray, PARAMS_WITH_BOUNDED_ARRAY_NUMPY
)
PARAMS_WITH_DENSITY_2D_NUMPY = (
    {
        "a": onp.asarray([1.0, 2.0, 3.0]),
        "b": (4.0, 5.0, 6.0),
    },
    onp.asarray([[7.0, 8.0]]),
    {
        "no_fixed_pixels": types.Density2DArray(
            array=onp.asarray([[9.0, 10.0, 11.0]]),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=None,
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=2,
        ),
        "fixed_solid": types.Density2DArray(
            array=onp.asarray([[12.0, 13.0, 14.0]]),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=onp.asarray([[1, 0, 0]], dtype=bool),
            fixed_void=None,
            minimum_width=1,
            minimum_spacing=2,
            periodic=(False, False),
        ),
        "fixed_void": types.Density2DArray(
            array=onp.asarray([[12.0, 13.0, 14.0]]),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=None,
            fixed_void=onp.asarray([[0, 0, 1]], dtype=bool),
            minimum_width=1,
            minimum_spacing=2,
        ),
        "fixed_solid_and_void": types.Density2DArray(
            array=onp.asarray([[12.0, 13.0, 14.0]]),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=onp.asarray([[1, 0, 0]], dtype=bool),
            fixed_void=onp.asarray([[0, 0, 1]], dtype=bool),
            minimum_width=1,
            minimum_spacing=2,
        ),
        "with_symmetries": types.Density2DArray(
            array=onp.asarray([[12.0, 13.0, 14.0]]),
            lower_bound=-1.0,
            upper_bound=1.0,
            fixed_solid=onp.asarray([[1, 0, 0]], dtype=bool),
            fixed_void=onp.asarray([[0, 0, 1]], dtype=bool),
            minimum_width=1,
            minimum_spacing=2,
            symmetries=(symmetry.REFLECTION_E_W, symmetry.ROTATION_180),
        ),
    },
)
PARAMS_WITH_DENSITY_2D_JAX = jax.tree_util.tree_map(
    jnp.asarray, PARAMS_WITH_DENSITY_2D_NUMPY
)
PARAMS = [
    PARAMS_SCALAR,
    PARAMS_BASIC,
    PARAMS_WITH_BOUNDED_ARRAY_NUMPY,
    PARAMS_WITH_BOUNDED_ARRAY_JAX,
    PARAMS_WITH_DENSITY_2D_NUMPY,
    PARAMS_WITH_DENSITY_2D_JAX,
]


def _lists_to_tuple(pytree, max_depth=10):
    for _ in range(max_depth):
        pytree = jax.tree_util.tree_map(
            lambda x: tuple(x) if isinstance(x, list) else x,
            pytree,
            is_leaf=lambda x: isinstance(x, list),
        )
    return pytree


def serialize(pytree) -> str:
    return json_utils.json_from_pytree(pytree=pytree)


def deserialize(serialized):
    return json_utils.pytree_from_json(serialized=serialized)


class BasicOptimizerTest(unittest.TestCase):
    @parameterized.parameterized.expand(itertools.product(PARAMS, OPTIMIZERS))
    def test_state_is_serializable(self, params, opt):
        state = opt.init(params)
        leaves, treedef = jax.tree_util.tree_flatten(state)

        serialized_state = serialize(state)
        restored_state = deserialize(serialized_state)
        # Serialization/deserialization unavoidably converts tuples to lists.
        # Convert back to tuples to facilitate comparison.
        restored_state = _lists_to_tuple(restored_state)
        restored_leaves, restored_treedef = jax.tree_util.tree_flatten(restored_state)

        self.assertEqual(treedef, restored_treedef)

        for a, b in zip(leaves, restored_leaves):
            onp.testing.assert_array_equal(a, b)

    @parameterized.parameterized.expand(itertools.product(PARAMS, OPTIMIZERS))
    def test_optimize(self, initial_params, opt):
        def loss_fn(params):
            leaves = jax.tree_util.tree_leaves(params)
            leaves_sum_squared = [jnp.sum(leaf**2) for leaf in leaves]
            return jnp.sum(jnp.asarray(leaves_sum_squared))

        state = opt.init(initial_params)
        for _ in range(3):
            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)

        initial_treedef = jax.tree_util.tree_structure(initial_params)
        treedef = jax.tree_util.tree_structure(params)
        # Assert that the tree structure (i.e. including auxilliary quantities) is
        # preserved by optimization.
        self.assertEqual(treedef, initial_treedef)

    @parameterized.parameterized.expand(itertools.product(PARAMS, OPTIMIZERS))
    def test_optimize_with_serialization(self, initial_params, opt):
        def loss_fn(params):
            leaves = jax.tree_util.tree_leaves(params)
            leaves_sum_squared = [jnp.sum(leaf**2) for leaf in leaves]
            return jnp.sum(jnp.asarray(leaves_sum_squared))

        # Optimize without serialization to get expected values.
        expected_params_list = []
        expected_value_list = []
        expected_grad_list = []
        state = opt.init(initial_params)
        for _ in range(3):
            params = opt.params(state)
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)
            expected_params_list.append(params)
            expected_value_list.append(value)
            expected_grad_list.append(grad)

        def serdes(x):
            return deserialize(serialize(x))

        # Optimize with serialization.
        params_list = []
        value_list = []
        grad_list = []
        state = opt.init(serdes(initial_params))
        for _ in range(3):
            state = serdes(state)
            params = serdes(opt.params(state))
            value, grad = jax.value_and_grad(loss_fn)(params)
            state = opt.update(
                grad=serdes(grad), value=serdes(value), params=params, state=state
            )
            params_list.append(params)
            value_list.append(value)
            grad_list.append(grad)

        self.assertSequenceEqual(value_list, expected_value_list)

        for p, ep in zip(params_list, expected_params_list):
            # Serialization/deserialization unavoidably converts tuples to lists.
            # Convert back to tuples to facilitate comparison.
            p = _lists_to_tuple(p)
            a_leaves, a_treedef = jax.tree_util.tree_flatten(p)
            b_leaves, b_treedef = jax.tree_util.tree_flatten(ep)
            self.assertEqual(a_treedef, b_treedef)
            for a, b in zip(a_leaves, b_leaves):
                onp.testing.assert_array_equal(a, b)

        for g, eg in zip(grad_list, expected_grad_list):
            g = _lists_to_tuple(g)
            a_leaves, a_treedef = jax.tree_util.tree_flatten(g)
            b_leaves, b_treedef = jax.tree_util.tree_flatten(eg)
            self.assertEqual(a_treedef, b_treedef)
            for a, b in zip(a_leaves, b_leaves):
                onp.testing.assert_array_equal(a, b)
