import jax.numpy as jnp

init_conds_kwargs = {
    "h_abl": jnp.array(200.0),
    "theta": jnp.array(288.0),
    "deltatheta": jnp.array(1.0),
    "wtheta": jnp.array(0.1),
    "q": jnp.array(0.008),
    "dq": jnp.array(-0.001),
    "co2": jnp.array(422.0),
    "deltaCO2": jnp.array(-44.0),
    "wCO2": jnp.array(0.0),
    "u": jnp.array(6.0),
    "du": jnp.array(4.0),
    "v": jnp.array(-4.0),
    "dv": jnp.array(4.0),
    "dz_h": jnp.array(150.0),
    "surf_pressure": jnp.array(101300.0),
}
""""""

model_kwargs = {
    "divU": jnp.array(0.0),
    "coriolis_param": jnp.array(1e-4),
    "gammatheta": jnp.array(0.006),
    "advtheta": jnp.array(0.0),
    "beta": jnp.array(0.2),
    "gammaq": jnp.array(0.0),
    "advq": jnp.array(0.0),
    "gammaCO2": jnp.array(0.0),
    "advCO2": jnp.array(0.0),
    "gammau": jnp.array(0.0),
    "advu": jnp.array(0.0),
    "gammav": jnp.array(0.0),
    "advv": jnp.array(0.0),
    "dFz": jnp.array(0.0),
    "is_shear_growing": True,
}
""""""
