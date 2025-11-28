# radiation #
std_rad_init_conds_kwargs = {
    "net_rad": 400,
}
std_rad_model_kwargs = {
    "lat": 51.97,
    "lon": -4.93,
    "doy": 268.0,
    "tstart": 6.8,
    "cc": 0.5,  # some cloud cover
}
""""""

# land surface #
ags_init_conds_kwargs = {
    "alpha": 0.18,  # darker, moist surface
    "wg": 0.35,  # moist topsoil
    "temp_soil": 285.0,
    "temp2": 285.5,
    "surf_temp": 287.0,
    "wl": 0.0001,  # dew or leaf water
}
""""""
ags_model_kwargs = {
    "a": 0.219,
    "b": 4.90,
    "p": 4.0,
    "cgsat": 3.56e-6,
    "wsat": 0.472,
    "wfc": 0.323,
    "wwilt": 0.171,
    "w2": 0.42,
    "d1": 0.5,
    "c1sat": 0.132,
    "c2ref": 1.8,
    "lai": 3.5,  # dense canopy
    "gD": 0.0,  # weak VPD dependence
    "rsmin": 60.0,  # strong stomatal conductance
    "rssoilmin": 30.0,  # low surface resistance
    "cveg": 0.95,  # nearly full cover
    "wmax": 0.0003,  # canopy water capacity
    "lam": 5.9,
    "c3c4": "c3",  # typical for temperate humid vegetation
}
""""""

# surface layer #
std_sl_init_conds_kwargs = {
    "ustar": 0.25,  # weaker turbulence (smooth, wet)
    "z0m": 0.03,
    "z0h": 0.003,
    "theta": 287.0,
}
""""""

# mixed layer #
bulk_ml_init_conds_kwargs = {
    "abl_height": 180.0,
    "theta": 287.0,
    "dtheta": 0.5,  # weak inversion
    "wtheta": 0.05,
    "q": 0.010,  # humid
    "dq": -0.0005,
    "wq": 2e-4,
    "co2": 420.0,
    "dCO2": -60.0,
    "wCO2": 0.0,
    "u": 4.0,
    "du": 2.0,
    "v": -3.0,
    "dv": 2.0,
    "dz_h": 120.0,
    "surf_pressure": 101300.0,
}
""""""
bulk_ml_model_kwargs = {
    "divU": 0.0,
    "coriolis_param": 1.0e-4,
    "gammatheta": 0.004,  # weaker lapse rate
    "advtheta": 0.0,
    "beta": 0.2,
    "gammaq": 0.0,
    "advq": 0.0,
    "gammaCO2": 0.0,
    "advCO2": 0.0,
    "gammau": 0.0,
    "advu": 0.0,
    "gammav": 0.0,
    "advv": 0.0,
    "dFz": 0.0,
    "sw_wind": True,
}
""""""
