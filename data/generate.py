import os

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from tqdm import tqdm

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import get_path_string

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
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    runtime = 12 * 3600.0
    tstart = 6.5

    # run integration
    times, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

    return trajectory, times


def sample_params(key):
    keys = random.split(key, 10)
    h_abl = random.uniform(keys[0], minval=50.0, maxval=400.0)
    theta = random.uniform(keys[1], minval=280.0, maxval=295.0)
    temp_noise = random.uniform(keys[2], minval=-2.0, maxval=2.0)
    temp_soil = theta + temp_noise
    q = random.uniform(keys[3], minval=0.004, maxval=0.012)
    deltatheta = random.uniform(keys[4], minval=0.5, maxval=2.0)
    u = random.uniform(keys[5], minval=2.0, maxval=12.0)
    v = random.uniform(keys[6], minval=-5.0, maxval=5.0)
    wg = random.uniform(keys[7], minval=0.171, maxval=0.35)
    d1 = random.uniform(keys[8], minval=0.1, maxval=1.0)
    cc = random.uniform(keys[9], minval=0.0, maxval=0.5)
    return h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = script_dir + "/dataset.h5"
    figs_dir = script_dir + "/../figs"
    os.makedirs(figs_dir, exist_ok=True)
    plot_file = figs_dir + "/statistics.png"

    key = random.PRNGKey(42)
    running_stats = {}
    times_template = None

    pbar = tqdm(total=NUM_TRAJS, desc="generating valid trajectories")
    valid_count = 0
    total_attempts = 0

    with h5py.File(output_file, "w") as f:
        datasets = {}

        while valid_count < NUM_TRAJS:
            total_attempts += 1
            key, subkey = random.split(key)

            # run simulation
            h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil = sample_params(
                subkey
            )
            traj_jax, times_jax = run_simulation(
                h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil
            )

            # convert to numpy for checking/saving
            trajectory = jax.tree.map(np.array, traj_jax)
            times = np.array(times_jax)

            if times_template is None:
                times_template = times
                f.create_dataset("time", data=times)

            # check for NaNs
            leaves, _ = jax.tree.flatten_with_path(trajectory)
            has_nan = False
            nan_vars = []

            for path, data in leaves:
                if np.isnan(data).any():
                    has_nan = True
                    var_name = get_path_string(path)
                    nan_vars.append(var_name)

            if has_nan:
                print(
                    f"[Warning] NaN detected in attempt {total_attempts}. "
                    f"Skipping. Variables affected: {nan_vars}"
                )
                continue

            # inits
            if valid_count == 0:
                timesteps = len(times)
                for path, data in leaves:
                    var_name = get_path_string(path)

                    # create HDF5 dataset
                    ds = f.create_dataset(
                        var_name,
                        shape=(NUM_TRAJS, timesteps),
                        dtype=data.dtype,
                    )
                    datasets[var_name] = ds

                    # init stats arrays
                    running_stats[var_name] = {
                        "sum": np.zeros(timesteps),
                        "sum_sq": np.zeros(timesteps),
                        "min": np.full(timesteps, np.inf),
                        "max": np.full(timesteps, -np.inf),
                    }

            # save data
            for path, data in leaves:
                var_name = get_path_string(path)
                datasets[var_name][valid_count, :] = data

                running_stats[var_name]["sum"] += data
                running_stats[var_name]["sum_sq"] += data**2
                running_stats[var_name]["min"] = np.minimum(
                    running_stats[var_name]["min"], data
                )
                running_stats[var_name]["max"] = np.maximum(
                    running_stats[var_name]["max"], data
                )

            valid_count += 1
            pbar.update(1)

        pbar.close()

        # finals stats calculation
        final_stats = {}
        for var_name, stats in running_stats.items():
            mean = stats["sum"] / NUM_TRAJS
            std = np.sqrt(np.clip(stats["sum_sq"] / NUM_TRAJS - mean**2, 0.0, None))
            final_stats[var_name] = {
                "mean": mean,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
            }

    print(
        f"saved dataset in {output_file} ({total_attempts} attempts for {NUM_TRAJS} valid samples)"
    )

    # plot to check if the trajectories are plausible
    plt.figure(figsize=(15, 10))
    plot_vars = [
        ("atmos/mixed/h_abl", "h [m]", 231),
        ("atmos/mixed/q", "q [kg/kg]", 232),
        ("land/hf", "H [W m-2]", 233),
        ("atmos/mixed/theta", "theta [K]", 234),
        ("atmos/clouds/cc_frac", "cloud fraction [-]", 235),
        ("land/le", "LE [W m-2]", 236),
    ]

    for var_name, ylabel, plot_id in plot_vars:
        if var_name not in final_stats:
            print(f"Warning: {var_name} not found in stats.")
            continue

        plt.subplot(plot_id)
        stats = final_stats[var_name]

        assert times_template is not None
        plt.plot(times_template, stats["mean"], label="mean")
        plt.fill_between(
            times_template,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
            label="±std",
        )
        plt.plot(times_template, stats["min"], "--", color="C0", label="min/max")
        plt.plot(times_template, stats["max"], "--", color="C0")

        plt.xlabel("time [h]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(var_name.split("/")[-1])

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"saved plot in {plot_file}")


if __name__ == "__main__":
    main()
