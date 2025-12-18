# rad #
std_rad_state_kwargs = {
    "net_rad": 400.0,
}
""""""
std_rad_model_kwargs = {
    "lat": 51.97,
    "lon": -4.93,
    "doy": 268.0,
    "cc": 0.0,
}

# land surface #
ags_state_kwargs = {
    "alpha": 0.25,
    "wg": 0.21,
    "temp_soil": 285.0,
    "temp2": 286.0,
    "surf_temp": 290.0,
    "wl": 0.0000,
    "wq": 1e-4,
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
    "w2": 0.21,
    "d1": 0.1,
    "c1sat": 0.132,
    "c2ref": 1.8,
    "lai": 2.0,
    "gD": 0.0,
    "rsmin": 110.0,
    "rssoilmin": 50.0,
    "cveg": 0.85,
    "wmax": 0.0002,
    "lam": 5.9,
    "c3c4": "c3",
}
""""""

# surface layer #
obukhov_sl_state_kwargs = {
    "ustar": 0.3,
    "z0m": 0.02,
    "z0h": 0.002,
}
""""""

# mixed layer #
bulk_ml_state_kwargs = {
    "h_abl": 200.0,
    "theta": 288.0,
    "deltatheta": 1.0,
    "wtheta": 0.1,
    "q": 0.008,
    "dq": -0.001,
    "co2": 422.0,
    "deltaCO2": -44.0,
    "wCO2": 0.0,
    "u": 6.0,
    "du": 4.0,
    "v": -4.0,
    "dv": 4.0,
    "dz_h": 150.0,
    "surf_pressure": 101300.0,
}
""""""

bulk_ml_model_kwargs = {
    "divU": 0.0,
    "coriolis_param": 1.0e-4,
    "gammatheta": 0.006,
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
