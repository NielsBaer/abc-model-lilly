# radiation #
std_rad_init_conds_kwargs = {
    "net_rad": 400,
}
""""""
std_rad_model_kwargs = {
    "lat": 51.97,
    "lon": -4.93,
    "doy": 268.0,
    "tstart": 6.8,
    "cc": 0.0,  # clear skies typical
}
""""""

# land surface #
ags_init_conds_kwargs = {
    "alpha": 0.30,  # higher albedo (dry soil, sparse cover)
    "wg": 0.10,  # low surface soil moisture
    "temp_soil": 293.0,  # warm soil
    "temp2": 292.0,
    "surf_temp": 297.0,  # hot surface
    "wl": 0.00002,  # minimal liquid water on leaves
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
    "w2": 0.12,
    "d1": 0.1,
    "c1sat": 0.132,
    "c2ref": 1.8,
    "lai": 0.8,  # sparse vegetation
    "gD": 0.1,  # strong VPD dependence
    "rsmin": 200.0,  # higher minimum stomatal resistance
    "rssoilmin": 150.0,  # dry surface resistance
    "cveg": 0.4,  # partial cover
    "wmax": 0.00005,  # thin water storage
    "lam": 5.9,
    "c3c4": "c4",  # C4-type adapted to arid conditions
}
""""""

# surface layer #
std_sl_init_conds_kwargs = {
    "ustar": 0.35,  # stronger mechanical mixing
    "z0m": 0.01,  # smoother dry soil
    "z0h": 0.001,
    "theta": 294.0,  # warm near-surface air
}
""""""

# mixed layer #
bulk_ml_init_conds_kwargs = {
    "h_abl": 250.0,
    "theta": 294.0,
    "deltatheta": 2.0,  # stronger inversion
    "wtheta": 0.1,
    "q": 0.004,  # low humidity
    "dq": -0.0015,
    "wq": 5e-5,
    "co2": 422.0,
    "deltaCO2": -20.0,
    "wCO2": 0.0,
    "u": 5.0,
    "du": 3.0,
    "v": -2.0,
    "dv": 3.0,
    "dz_h": 150.0,
    "surf_pressure": 101000.0,
}
""""""
bulk_ml_model_kwargs = {
    "divU": 0.0,
    "coriolis_param": 1.0e-4,
    "gammatheta": 0.008,  # stronger stability
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
}
""""""
