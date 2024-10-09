"""Defines tests for the `lbfgsb.lbfgsb` module.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse

import jax
import jax.numpy as jnp
from jax import flatten_util
from totypes import types

from invrs_opt.optimizers import lbfgsb

jax.config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser(prog="debug", description="opt debugging")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of steps.",
)

if __name__ == "__main__":

    print("running")

    args = parser.parse_args()
    steps = args.steps

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

    print("state initialized")

    @jax.jit
    @jax.vmap
    def step_fn(state):
        params = opt.params(state)
        value, grad = jax.value_and_grad(loss_fn)(params)
        state = opt.update(grad=grad, value=value, params=params, state=state)
        return state, value

    for i in range(steps):
        print(f"batch ({i})")
        state, value = step_fn(state)

    print("batch results complete")

    # Test one-at-a-time optimization.
    for k in keys:
        params = initial_params_fn(k)
        state = opt.init(params)
        for i in range(steps):
            print(f"one-at-a-time ({i}/{k})")
            params = opt.params(state)
            value, grad = jax.jit(jax.value_and_grad(loss_fn))(params)
            state = opt.update(grad=grad, value=value, params=params, state=state)

    print("one-at-a-time results complete")
