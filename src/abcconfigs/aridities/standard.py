import jax.numpy as jnp

# rad #
std_rad_init_conds_kwargs = {
    "net_rad": jnp.array(400.0),
}
""""""
std_rad_model_kwargs = {
    "lat": jnp.array(51.97),
    "lon": jnp.array(-4.93),
    "doy": jnp.array(268.0),
    "tstart": jnp.array(6.8),
    "cc": jnp.array(0.0),
}

# land surface #
ags_init_conds_kwargs = {
    "alpha": jnp.array(0.25),
    "wg": jnp.array(0.21),
    "temp_soil": jnp.array(285.0),
    "temp2": jnp.array(286.0),
    "surf_temp": jnp.array(290.0),
    "wl": jnp.array(0.0000),
}
""""""
ags_model_kwargs = {
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

# surface layer #
std_sl_init_conds_kwargs = {
    "ustar": jnp.array(0.3),
    "z0m": jnp.array(0.02),
    "z0h": jnp.array(0.002),
    "theta": jnp.array(288.0),
}
""""""

# mixed layer #
bulk_ml_init_conds_kwargs = {
    "h_abl": jnp.array(200.0),
    "theta": jnp.array(288.0),
    "deltatheta": jnp.array(1.0),
    "wtheta": jnp.array(0.1),
    "q": jnp.array(0.008),
    "dq": jnp.array(-0.001),
    "wq": jnp.array(1e-4),
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

bulk_ml_model_kwargs = {
    "divU": jnp.array(0.0),
    "coriolis_param": jnp.array(1.0e-4),
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
}
""""""
