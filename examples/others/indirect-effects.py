import jax.numpy as jnp
import matplotlib.pyplot as plt

import abcconfigs.aridities as aridity_config
import abcmodel
from abcmodel.utils import compute_esat


def run_wrapper(wg: float, q: float, config):
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # rad with clouds
    rad_init_conds = abcmodel.rad.StandardRadiationInitConds(
        **config.std_rad_init_conds_kwargs
    )
    rad_model = abcmodel.rad.StandardRadiationModel(**config.std_rad_model_kwargs)

    # land surface
    ags_kwargs = config.ags_init_conds_kwargs
    ags_kwargs["wg"] = wg
    land_init_conds = abcmodel.land.AgsInitConds(
        **ags_kwargs,
    )
    land_model = abcmodel.land.AgsModel(
        **config.ags_model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = (
        abcmodel.atmos.surface_layer.ObukhovSurfaceLayerInitConds(
            **config.std_sl_init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()

    # mixed layer
    ml_kwargs = config.bulk_ml_init_conds_kwargs
    ml_kwargs["q"] = q
    mixed_layer_init_conds = abcmodel.atmos.mixed_layer.BulkMixedLayerInitConds(
        **ml_kwargs,
    )
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **config.bulk_ml_model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.atmos.clouds.CumulusInitConds()
    cloud_model = abcmodel.atmos.clouds.CumulusModel()

    # define atmos model
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_init_conds = abcmodel.atmos.DayOnlyAtmosphereState(
        surface=surface_layer_init_conds,
        mixed=mixed_layer_init_conds,
        clouds=cloud_init_conds,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )

    state = abcoupler.init_state(
        rad_init_conds,
        land_init_conds,
        atmos_init_conds,
    )

    return abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)


def make_fancy_plot(
    axes,
    time: jnp.ndarray,
    traj,
    color: str,
    marker,
    label: str,
    factor: int = 30,
):
    # mixed layer
    axes[0, 0].plot(
        time[::factor],
        traj.atmos.mixed_layer.h_abl[::factor],
        color=color,
        marker=marker,
        linestyle="None",
        label=f"{label}",
    )
    axes[0, 0].set_title("h [m]")

    axes[0, 1].plot(
        time[::factor],
        traj.land.wCO2A[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 1].set_title("wCO2A [mgC/m²/s]")

    axes[0, 2].plot(
        time[::factor],
        traj.land.wCO2R[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 2].set_title("wCO2R [mgC/m²/s]")

    axes[0, 3].plot(
        time[::factor],
        traj.land.wCO2[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 3].set_title("wCO2 [mgC/m²/s]")

    # temperature
    axes[1, 0].plot(
        time[::factor],
        traj.land.temp_soil[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 0].set_title("temp soil [K]")

    axes[1, 1].plot(
        time[::factor],
        traj.land.surf_temp[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 1].set_title("surf temp [K]")

    axes[1, 2].plot(
        time[::factor],
        traj.atmos.mixed_layer.theta[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 2].set_title("theta [K]")

    axes[1, 3].plot(
        time[::factor],
        traj.atmos.surface_layer.temp_2m[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 3].set_title("temp 2m [K]")

    # water
    # this is the core of this example! #
    vpd = (compute_esat(traj.land.surf_temp) - traj.land.e) / 1000.0  # kPa
    axes[2, 0].plot(
        time[::factor], vpd[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[2, 0].set_title("VPD [kPa]")

    axes[2, 1].plot(
        time[::factor],
        traj.land.wg[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[2, 1].set_title("wg [kg/kg]")
    # - - - - - - - - - - - - - - - - - #
    axes[2, 2].plot(
        time[::factor],
        traj.atmos.mixed_layer.q[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[2, 2].set_title("q [kg/kg]")

    # axes[2, 3] is dedicated to legend display only
    axes[2, 3].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[2, 3].legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        fontsize=12,
        handlelength=2,
    )

    # energy fluxes
    axes[3, 0].plot(
        time[::factor],
        traj.land.hf[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 0].set_title("H [W/m²]")

    axes[3, 1].plot(
        time[::factor],
        traj.land.le[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 1].set_title("LE [W/m²]")

    axes[3, 2].plot(
        time[::factor],
        traj.land.le_veg[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 2].set_title("LEveg [W/m²]")

    axes[3, 3].plot(
        time[::factor],
        traj.land.le_liq[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 3].set_title("LEliq [W/m²]")

    # rad
    axes[4, 0].plot(
        time[::factor],
        traj.rad.in_srad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 0].set_title("SWin [W/m²]")

    axes[4, 1].plot(
        time[::factor],
        traj.rad.out_srad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 1].set_title("SWout [W/m²]")

    axes[4, 2].plot(
        time[::factor],
        traj.rad.in_lrad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 2].set_title("LWin [W/m²]")

    axes[4, 3].plot(
        time[::factor],
        traj.rad.out_lrad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 3].set_title("LWout [W/m²]")


def run_experiment(
    wg_control: float,
    q_control: float,
    delta_wg: float,
    delta_q: float,
    config,
    exp: str,
):
    # plot
    fig, axes = plt.subplots(5, 4, figsize=(18, 9))

    # add experiments:
    time, control_traj = run_wrapper(wg_control, q_control, config)
    make_fancy_plot(axes, time, control_traj, "gray", "o", f"{exp} control")
    # positive soil moisture anomaly
    time, pos_soil_moist_traj = run_wrapper(wg_control + delta_wg, q_control, config)
    make_fancy_plot(axes, time, pos_soil_moist_traj, "dodgerblue", "+", "+|deltaSM|")
    # negative soil moisture anomaly
    time, neg_soil_moist_traj = run_wrapper(wg_control - delta_wg, q_control, config)
    make_fancy_plot(axes, time, neg_soil_moist_traj, "dodgerblue", "_", "-|deltaSM|")
    # positive specific humidity anomaly
    time, pos_surf_temp_traj = run_wrapper(wg_control, q_control + delta_q, config)
    make_fancy_plot(axes, time, pos_surf_temp_traj, "orangered", "+", "+|deltaq|")
    # negative specific humidity anomaly
    time, neg_surf_temp_traj = run_wrapper(wg_control, q_control - delta_q, config)
    make_fancy_plot(axes, time, neg_surf_temp_traj, "orangered", "_", "-|deltaq|")

    # names on each row
    row_names = [
        "Mixed Layer",
        "Temperature",
        "Water",
        "Energy Fluxes",
        "Radiation",
    ]
    for i, name in enumerate(row_names):
        ax = axes[i, 0]
        ax.set_ylabel(
            name,
            fontweight="bold",
        )

    # only put x-axis labels and ticks on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [h]")
    for ax in axes[:-1, :].flatten():
        ax.set_xticks([])

    # leave space for row labels
    plt.tight_layout()
    plt.show()

    return fig


def main():
    # standard experiment
    wg_control = 0.21
    delta_wg = 0.105
    q_control = 0.008
    delta_q = 0.004
    config = aridity_config.standard
    run_experiment(wg_control, q_control, delta_wg, delta_q, config, "standard")

    # arid experiment
    wg_control = 0.1
    delta_wg = 0.05
    q_control = 0.004
    delta_q = 0.002
    config = aridity_config.arid
    run_experiment(wg_control, q_control, delta_wg, delta_q, config, "arid")

    # humid experiment
    wg_control = 0.35
    delta_wg = 0.175
    q_control = 0.01
    delta_q = 0.005
    config = aridity_config.humid
    run_experiment(wg_control, q_control, delta_wg, delta_q, config, "humid")


if __name__ == "__main__":
    main()
