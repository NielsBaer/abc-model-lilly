import jax.numpy as jnp

# rad #
std_rad_init_conds_kwargs = {
    "net_rad": jnp.array(400),
}
""""""
std_rad_model_kwargs = {
    "lat": jnp.array(51.97),
    "lon": jnp.array(-4.93),
    "doy": jnp.array(268.0),
    "tstart": jnp.array(6.8),
    "cc": jnp.array(0.0),  # clear skies typical
}
""""""

# land surface #
ags_init_conds_kwargs = {
    "alpha": jnp.array(0.30),  # higher albedo (dry soil, sparse cover)
    "wg": jnp.array(0.10),  # low surface soil moisture
    "temp_soil": jnp.array(293.0),  # warm soil
    "temp2": jnp.array(292.0),
    "surf_temp": jnp.array(297.0),  # hot surface
    "wl": jnp.array(0.00002),  # minimal liquid water on leaves
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
    "w2": jnp.array(0.12),
    "d1": jnp.array(0.1),
    "c1sat": jnp.array(0.132),
    "c2ref": jnp.array(1.8),
    "lai": jnp.array(0.8),  # sparse vegetation
    "gD": jnp.array(0.1),  # strong VPD dependence
    "rsmin": jnp.array(200.0),  # higher minimum stomatal resistance
    "rssoilmin": jnp.array(150.0),  # dry surface resistance
    "cveg": jnp.array(0.4),  # partial cover
    "wmax": jnp.array(0.00005),  # thin water storage
    "lam": jnp.array(5.9),
    "c3c4": "c4",  # C4-type adapted to arid conditions
}
""""""

# surface layer #
std_sl_init_conds_kwargs = {
    "ustar": jnp.array(0.35),  # stronger mechanical mixing
    "z0m": jnp.array(0.01),  # smoother dry soil
    "z0h": jnp.array(0.001),
    "theta": jnp.array(294.0),  # warm near-surface air
}
""""""

# mixed layer #
bulk_ml_init_conds_kwargs = {
    "h_abl": jnp.array(250.0),
    "theta": jnp.array(294.0),
    "deltatheta": jnp.array(2.0),  # stronger inversion
    "wtheta": jnp.array(0.1),
    "q": jnp.array(0.004),  # low humidity
    "dq": jnp.array(-0.0015),
    "wq": jnp.array(5e-5),
    "co2": jnp.array(422.0),
    "deltaCO2": jnp.array(-20.0),
    "wCO2": jnp.array(0.0),
    "u": jnp.array(5.0),
    "du": jnp.array(3.0),
    "v": jnp.array(-2.0),
    "dv": jnp.array(3.0),
    "dz_h": jnp.array(150.0),
    "surf_pressure": jnp.array(101000.0),
}
""""""
bulk_ml_model_kwargs = {
    "divU": jnp.array(0.0),
    "coriolis_param": jnp.array(1.0e-4),
    "gammatheta": jnp.array(0.008),  # stronger stability
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
