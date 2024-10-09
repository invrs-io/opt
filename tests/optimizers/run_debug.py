"""Defines tests for the `lbfgsb.lbfgsb` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse

import jax
import jax.numpy as jnp
from totypes import types

from invrs_opt.optimizers import lbfgsb

jax.config.update("jax_enable_x64", True)


def optimization_with_vmap(steps):

    print("running", flush=True)

    def initial_params_fn(key):
        ka, kb = jax.random.split(key)
        return {
            "a": jax.random.normal(ka, (10,)),
            "b": jax.random.normal(kb, (10,)),
            "c": types.Density2DArray(array=jnp.ones((3, 3))),
        }

    keys = jax.random.split(jax.random.PRNGKey(0))
    opt = lbfgsb.density_lbfgsb(beta=2, maxcor=20)

    # Test batch optimization
    params = jax.vmap(initial_params_fn)(keys)
    state = jax.vmap(opt.init)(params)

    print("state initialized", flush=True)

    @jax.jit
    @jax.vmap
    def step_fn(state):
        params = opt.params(state)
        dummy_value = jnp.array(1.0, dtype=float)
        dummy_grad = jax.tree_util.tree_map(jnp.ones_like, params)
        state = opt.update(grad=dummy_grad, value=dummy_value, params=params, state=state)
        return state, dummy_value

    for i in range(steps):
        print(f"batch ({i})", flush=True)
        state, value = step_fn(state)

    # Test one-at-a-time optimization.
    for k in keys:
        print(f"key={k}", flush=True)
        params = initial_params_fn(k)
        print("params initialized", flush=True)
        state = opt.init(params)
        print("state initialized", flush=True)
        for i in range(steps):
            print(f"one-at-a-time ({i}/{k})", flush=True)
            params = opt.params(state)
            dummy_value = jnp.array(1.0, dtype=float)
            dummy_grad = jax.tree_util.tree_map(jnp.ones_like, params)
            state = opt.update(grad=dummy_grad, value=dummy_value, params=params, state=state)

    print("one-at-a-time results complete", flush=True)


parser = argparse.ArgumentParser(prog="debug", description="opt debugging")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of steps.",
)

if __name__ == "__main__":

    args = parser.parse_args()
    optimization_with_vmap(steps=args.steps)
