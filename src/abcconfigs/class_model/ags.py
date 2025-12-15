import jax.numpy as jnp

init_conds_kwargs = {
    "alpha": jnp.array(0.25),
    "wg": jnp.array(0.21),
    "temp_soil": jnp.array(285.0),
    "temp2": jnp.array(286.0),
    "surf_temp": jnp.array(290.0),
    "wl": jnp.array(0.0000),
    "wq": jnp.array(1e-4),
}
""""""
model_kwargs = {
    "a": jnp.array(0.219),
    "b": jnp.array(4.90),
    "p": jnp.array(4.0),
    "cgsat": jnp.array(3.56e-6),
    "wsat": jnp.array(0.472),
    "wfc": jnp.array(0.323),
    "wwilt": jnp.array(0.171),
    "w2": jnp.array(0.21),
    "d1": jnp.array(0.1),
    "c1sat": jnp.array(0.132),
    "c2ref": jnp.array(1.8),
    "lai": jnp.array(2.0),
    "gD": jnp.array(0.0),
    "rsmin": jnp.array(110.0),
    "rssoilmin": jnp.array(50.0),
    "cveg": jnp.array(0.85),
    "wmax": jnp.array(0.0002),
    "lam": jnp.array(5.9),
    "c3c4": "c3",
}
""""""
