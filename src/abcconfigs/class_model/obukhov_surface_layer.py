import jax.numpy as jnp

init_conds_kwargs = {
    "ustar": jnp.array(0.3),
    "z0m": jnp.array(0.02),
    "z0h": jnp.array(0.002),
    "theta": jnp.array(288.0),
}
"""The model takes no parameters!"""
