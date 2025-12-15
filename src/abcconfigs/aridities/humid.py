import jax.numpy as jnp

# rad #
std_rad_init_conds_kwargs = {
    "net_rad": jnp.array(400.0),
}
std_rad_model_kwargs = {
    "lat": jnp.array(51.97),
    "lon": jnp.array(-4.93),
    "doy": jnp.array(268.0),
    "tstart": jnp.array(6.8),
    "cc": jnp.array(0.5),  # some cloud cover
}
""""""

# land surface #
ags_init_conds_kwargs = {
    "alpha": jnp.array(0.18),  # darker, moist surface
    "wg": jnp.array(0.35),  # moist topsoil
    "temp_soil": jnp.array(285.0),
    "temp2": jnp.array(285.5),
    "surf_temp": jnp.array(287.0),
    "wl": jnp.array(0.0001),  # dew or leaf water
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
    "w2": jnp.array(0.42),
    "d1": jnp.array(0.5),
    "c1sat": jnp.array(0.132),
    "c2ref": jnp.array(1.8),
    "lai": jnp.array(3.5),  # dense canopy
    "gD": jnp.array(0.0),  # weak VPD dependence
    "rsmin": jnp.array(60.0),  # strong stomatal conductance
    "rssoilmin": jnp.array(30.0),  # low surface resistance
    "cveg": jnp.array(0.95),  # nearly full cover
    "wmax": jnp.array(0.0003),  # canopy water capacity
    "lam": jnp.array(5.9),
    "c3c4": "c3",  # typical for temperate humid vegetation
}
""""""

# surface layer #
std_sl_init_conds_kwargs = {
    "ustar": jnp.array(0.25),  # weaker turbulence (smooth, wet)
    "z0m": jnp.array(0.03),
    "z0h": jnp.array(0.003),
    "theta": jnp.array(287.0),
}
""""""

# mixed layer #
bulk_ml_init_conds_kwargs = {
    "h_abl": jnp.array(180.0),
    "theta": jnp.array(287.0),
    "deltatheta": jnp.array(0.5),  # weak inversion
    "wtheta": jnp.array(0.05),
    "q": jnp.array(0.010),  # humid
    "dq": jnp.array(-0.0005),
    "wq": jnp.array(2e-4),
    "co2": jnp.array(420.0),
    "deltaCO2": jnp.array(-60.0),
    "wCO2": jnp.array(0.0),
    "u": jnp.array(4.0),
    "du": jnp.array(2.0),
    "v": jnp.array(-3.0),
    "dv": jnp.array(2.0),
    "dz_h": jnp.array(120.0),
    "surf_pressure": jnp.array(101300.0),
}
""""""
bulk_ml_model_kwargs = {
    "divU": jnp.array(0.0),
    "coriolis_param": jnp.array(1.0e-4),
    "gammatheta": jnp.array(0.004),  # weaker lapse rate
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
