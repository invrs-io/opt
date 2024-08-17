# invrs-opt - Optimization algorithms for inverse design
`v0.9.1`

## Overview

The `invrs-opt` package defines an optimizer API intended for topology optimization, inverse design, or AI-guided design. It (currently) implements the L-BFGS-B optimization algorithm along with some variants. The API is intended to be general so that new algorithms can be accommodated, and is inspired by the functional optimizer approach used in jax. Example usage is as follows:

```python
initial_params = ...

optimizer = invrs_opt.lbfgsb()
state = optimizer.init(initial_params)

for _ in range(steps):
    params = optimizer.params(state)
    value, grad = jax.value_and_grad(loss_fn)(params)
    state = optimizer.update(grad=grad, value=value, params=params, state=state)
```

Optimizers in `invrs-opt` are compatible with custom types defined in the [totypes](https://github.com/invrs-io/totypes) package. The basic `lbfgsb` optimizer enforces bounds for custom types, while the `density_lbfgsb` optimizer implements a filter-and-threshold operation for `DensityArray2D` types to ensure that solutions have the correct length scale.

## Install
```
pip install invrs_opt
```
