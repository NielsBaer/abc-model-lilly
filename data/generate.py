from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from tqdm import tqdm

import abcconfigs.class_model as cm
import abcmodel

NUM_TRAJS = 100


@jax.jit
def run_simulation(h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil):
    # rad
    rad_model_kwargs = cm.standard_radiation.model_kwargs
    rad_model_kwargs["cc"] = cc
    rad_model = abcmodel.rad.StandardRadiationModel(**rad_model_kwargs)
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # land
    ags_model_kwargs = cm.ags.model_kwargs
    ags_model_kwargs["d1"] = d1
    land_model = abcmodel.land.AgsModel(**ags_model_kwargs)

    ags_state_kwargs = cm.ags.state_kwargs
    ags_state_kwargs["wg"] = wg
    ags_state_kwargs["temp_soil"] = temp_soil
    land_state = land_model.init_state(**ags_state_kwargs)

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer
    mixed_state_kwargs = cm.bulk_mixed_layer.state_kwargs
    mixed_state_kwargs.update(
        {
            "h_abl": h_abl,
            "theta": theta,
            "deltatheta": deltatheta,
            "q": q,
            "u": u,
            "v": v,
        }
    )
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    mixed_layer_state = mixed_layer_model.init_state(
        **mixed_state_kwargs,
    )

    # clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # atmosphere
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_state,
        mixed=mixed_layer_state,
        clouds=cloud_state,
    )

    # coupler
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(
        rad_state,
        land_state,
        atmos_state,
    )

    # integration settings
    dt = 15.0
    runtime = 12 * 3600.0
    tstart = 6.5

    # run integration
    times, trajectory = abcmodel.integrate(state, abcoupler, dt, runtime, tstart)

    # extract core variables to save
    output = {
        "cc_frac": trajectory.atmos.clouds.cc_frac,
        "cc_mf": trajectory.atmos.clouds.cc_mf,
        "cc_qf": trajectory.atmos.clouds.cc_qf,
        "cl_trans": trajectory.atmos.clouds.cl_trans,
        "h_abl": trajectory.atmos.mixed.h_abl,
        "theta": trajectory.atmos.mixed.theta,
        "q": trajectory.atmos.mixed.q,
        "co2": trajectory.atmos.mixed.co2,
        "ustar": trajectory.atmos.surface.ustar,
        "thetasurf": trajectory.atmos.surface.thetasurf,
        "hf": trajectory.land.hf,
        "le": trajectory.land.le,
        "surf_temp": trajectory.land.surf_temp,
        "wg": trajectory.land.wg,
        "in_srad": trajectory.rad.in_srad,
        "out_srad": trajectory.rad.out_srad,
        "in_lrad": trajectory.rad.in_lrad,
        "out_lrad": trajectory.rad.out_lrad,
    }

    return output, times


def sample_params(key):
    keys = random.split(key, 10)

    # perturb initial conditions and parameters
    h_abl = random.uniform(keys[0], minval=50.0, maxval=400.0)
    theta = random.uniform(keys[1], minval=280.0, maxval=295.0)

    # correlation between soil temp and air temp
    temp_noise = random.uniform(keys[2], minval=-2.0, maxval=2.0)
    temp_soil = theta + temp_noise

    q = random.uniform(keys[3], minval=0.004, maxval=0.012)
    deltatheta = random.uniform(keys[4], minval=0.5, maxval=2.0)

    # winds
    u = random.uniform(keys[5], minval=2.0, maxval=12.0)
    v = random.uniform(keys[6], minval=-5.0, maxval=5.0)

    # soil
    wg = random.uniform(keys[7], minval=0.171, maxval=0.35)
    d1 = random.uniform(keys[8], minval=0.1, maxval=1.0)

    # cloud cover
    cc = random.uniform(keys[9], minval=0.0, maxval=0.5)

    return h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil


def main():
    # setup paths relative to script location
    script_dir = Path(__file__).parent.resolve()
    output_file = script_dir / "dataset.h5"
    figs_dir = script_dir.parent / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_file = figs_dir / "statistics.png"

    key = random.PRNGKey(42)
    running_stats = {}
    times_template = None

    with h5py.File(output_file, "w") as f:
        grp_raw = f.create_group("raw")
        datasets = {}

        for i in tqdm(range(NUM_TRAJS), desc="generating trajectories"):
            key, subkey = random.split(key)

            # sample parameters following distributions
            h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil = sample_params(
                subkey
            )
            results_jax, times_jax = run_simulation(
                h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil
            )

            # from jax to numpy for saving efficiency
            results = jax.tree.map(np.array, results_jax)
            times = np.array(times_jax)

            # save time once
            if times_template is None:
                times_template = times
                f.create_dataset("time", data=times)

            # on first iteration, create datasets and init stats
            if i == 0:
                timesteps = len(times)
                for var_name, data in results.items():
                    # create dataset and keep a handle to it to avoid ambiguous indexing types
                    ds = grp_raw.create_dataset(
                        var_name,
                        shape=(NUM_TRAJS, timesteps),
                        dtype=data.dtype,
                    )
                    datasets[var_name] = ds

                    running_stats[var_name] = {
                        "sum": np.zeros(timesteps),
                        "sum_sq": np.zeros(timesteps),
                        "min": np.full(timesteps, np.inf),
                        "max": np.full(timesteps, -np.inf),
                    }

            # save results
            for var_name, data in results.items():
                ds = datasets[var_name]
                # assign into dataset explicitly using slice indexing
                ds[i, :] = data

                # update stats
                running_stats[var_name]["sum"] += data
                running_stats[var_name]["sum_sq"] += data**2
                running_stats[var_name]["min"] = np.minimum(
                    running_stats[var_name]["min"], data
                )
                running_stats[var_name]["max"] = np.maximum(
                    running_stats[var_name]["max"], data
                )

            # memory cleanup (needed???)
            del results_jax
            del times_jax
            del results

        final_stats = {}

        for var_name, stats in running_stats.items():
            mean = stats["sum"] / NUM_TRAJS
            std = np.sqrt(stats["sum_sq"] / NUM_TRAJS - mean**2)
            final_stats[var_name] = {
                "mean": mean,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
            }

    print(f"saved dataset in {output_file}")

    plt.figure(figsize=(15, 10))
    plot_vars = [
        ("h_abl", "h [m]", 231),
        ("q", "q [kg/kg]", 232),
        ("hf", "H [W m-2]", 233),
        ("theta", "theta [K]", 234),
        ("cc_frac", "cloud fraction [-]", 235),
        ("le", "LE [W m-2]", 236),
    ]

    for var_name, ylabel, plot_id in plot_vars:
        if var_name not in final_stats:
            continue

        plt.subplot(plot_id)

        stats = final_stats[var_name]
        mean = stats["mean"]
        std = stats["std"]
        lower_bound = mean - std
        upper_bound = mean + std
        min_val = stats["min"]
        max_val = stats["max"]

        assert times_template is not None
        plt.plot(times_template, mean, label="mean")
        plt.fill_between(
            times_template,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="±std",
        )
        plt.plot(times_template, min_val, "--", label="min", color="C0")
        plt.plot(times_template, max_val, "--", label="max", color="C0")
        plt.xlabel("time [h]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(var_name)

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"saved plot in {plot_file}")


if __name__ == "__main__":
    main()
