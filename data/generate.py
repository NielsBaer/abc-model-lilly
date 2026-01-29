import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import os
import gc

import abcconfigs.class_model as cm
import abcmodel

@jax.jit
def run_simulation(
    h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil
):
    # Initialize models and states
    # Rad
    rad_model_kwargs = cm.standard_radiation.model_kwargs.copy()
    rad_model_kwargs["cc"] = cc
    rad_model = abcmodel.rad.StandardRadiationModel(**rad_model_kwargs)
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # Land
    ags_model_kwargs = cm.ags.model_kwargs.copy()
    ags_model_kwargs["d1"] = d1
    land_model = abcmodel.land.AgsModel(**ags_model_kwargs)
    
    ags_state_kwargs = cm.ags.state_kwargs.copy()
    ags_state_kwargs["wg"] = wg
    ags_state_kwargs["temp_soil"] = temp_soil
    land_state = land_model.init_state(**ags_state_kwargs)

    # Surface Layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # Mixed Layer
    mixed_state_kwargs = cm.bulk_mixed_layer.state_kwargs.copy()
    mixed_state_kwargs.update({
        "h_abl": h_abl,
        "theta": theta,
        "deltatheta": deltatheta,
        "q": q,
        "u": u,
        "v": v,
    })
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    mixed_layer_state = mixed_layer_model.init_state(
        **mixed_state_kwargs,
    )

    # Clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # Atmosphere
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

    # Coupler
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

    # Integration settings
    dt = 15.0
    runtime = 12 * 3600.0
    tstart = 6.8

    # Run integration
    times, trajectory = abcmodel.integrate(state, abcoupler, dt, runtime, tstart)

    # Extract core variables
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
    # Split keys for parameters
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = random.split(key, 9)

    # Perturb initial conditions and parameters
    h_abl = random.uniform(k1, minval=50.0, maxval=400.0)
    theta = random.uniform(k2, minval=280.0, maxval=295.0)
    
    # Correlation between soil temp and air temp
    temp_noise = random.uniform(k3, minval=-2.0, maxval=2.0)
    temp_soil = theta + temp_noise
    
    q = random.uniform(k4, minval=0.004, maxval=0.012)
    deltatheta = random.uniform(k5, minval=0.5, maxval=2.0)
    
    # Winds
    u = random.uniform(k6, minval=2.0, maxval=12.0)
    v = random.uniform(k7, minval=-5.0, maxval=5.0)
    
    # Soil
    wg = random.uniform(k8, minval=0.171, maxval=0.35)
    d1 = random.uniform(k9, minval=0.1, maxval=1.0)
    
    # Cloud cover
    k10, _ = random.split(k9)
    cc = random.uniform(k10, minval=0.0, maxval=0.5)
    
    return h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil

def main():
    NUM_TRAJS = 100
    
    output_file = "data/generated_dataset.h5"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"Generating {NUM_TRAJS} trajectories fully sequentially...")
    
    # Master key
    key = random.PRNGKey(42)
    
    running_stats = {}
    times_template = None
    
    with h5py.File(output_file, "w") as f:
        grp_raw = f.create_group("raw")
        
        for i in range(NUM_TRAJS):
            print(f"  Trajectory {i + 1}/{NUM_TRAJS}...")
            
            # Prepare key
            key, subkey = random.split(key)
            
            # Sample parameters
            h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil = sample_params(subkey)
            
            # Debugging: Print parameters
            print(f"    Params: h_abl={h_abl:.2f}, theta={theta:.2f}, q={q:.5f}, dt={deltatheta:.2f}, u={u:.2f}, v={v:.2f}, wg={wg:.3f}, d1={d1:.3f}, cc={cc:.2f}, t_soil={temp_soil:.2f}")
            
            # Run simulation
            try:
                # We need to pass the parameters to run_simulation
                results_jax, times_jax = run_simulation(
                    h_abl, theta, q, deltatheta, u, v, wg, d1, cc, temp_soil
                )
                
                # Block until ready to ensure we catch errors here
                results_jax["h_abl"].block_until_ready()
                
            except Exception as e:
                print(f"    ERROR in trajectory {i+1}: {e}")
                continue  # Skip this trajectory
            
            # Transfer to host
            results = jax.tree.map(np.array, results_jax)
            times = np.array(times_jax)
            
            if times_template is None:
                times_template = times
                f.create_dataset("time", data=times)
            
            # On first iteration, create datasets and init stats
            if i == 0:
                timesteps = len(times)
                for var_name, data in results.items():
                    grp_raw.create_dataset(
                        var_name, 
                        shape=(NUM_TRAJS, timesteps), 
                        dtype=data.dtype,
                    )
                    
                    running_stats[var_name] = {
                        "sum": np.zeros(timesteps),
                        "sum_sq": np.zeros(timesteps),
                        "min": np.full(timesteps, np.inf),
                        "max": np.full(timesteps, -np.inf)
                    }
            
            # Save results
            for var_name, data in results.items():
                grp_raw[var_name][i] = data
                
                # Update stats
                running_stats[var_name]["sum"] += data
                running_stats[var_name]["sum_sq"] += data**2
                running_stats[var_name]["min"] = np.minimum(running_stats[var_name]["min"], data)
                running_stats[var_name]["max"] = np.maximum(running_stats[var_name]["max"], data)
                
            # Memory cleanup
            del results_jax
            del times_jax
            del results
            gc.collect() # Explicitly request GC

        # Compute final stats
        print("Computing final statistics...")
        grp_stats = f.create_group("statistics")
        
        final_stats = {}
        
        for var_name, stats in running_stats.items():
            mean = stats["sum"] / NUM_TRAJS
            variance = (stats["sum_sq"] / NUM_TRAJS) - (mean**2)
            std = np.sqrt(np.maximum(variance, 0))
            
            subgrp = grp_stats.create_group(var_name)
            subgrp.create_dataset("mean", data=mean)
            subgrp.create_dataset("std", data=std)
            subgrp.create_dataset("min", data=stats["min"])
            subgrp.create_dataset("max", data=stats["max"])
            
            final_stats[var_name] = {
                "mean": mean,
                "min": stats["min"],
                "max": stats["max"]
            }

    print(f"Saved to {output_file}")

    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(15, 10))
    
    plot_vars = [
        ("h_abl", "h [m]", 231),
        ("q", "q [kg/kg]", 232),
        ("hf", "Sensible Heat Flux [W m-2]", 233),
        ("theta", "theta [K]", 234),
        ("cc_frac", "Cloud Fraction [-]", 235),
        ("le", "Latent Heat Flux [W m-2]", 236),
    ]

    for var_name, ylabel, plot_id in plot_vars:
        if var_name not in final_stats:
            continue
            
        plt.subplot(plot_id)
        
        stats = final_stats[var_name]
        mean = stats["mean"]
        min_val = stats["min"]
        max_val = stats["max"]
        
        plt.plot(times_template, mean, 'b-', label='Mean')
        plt.fill_between(times_template, min_val, max_val, color='b', alpha=0.2, label='Range')
        
        plt.xlabel("time [h]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(var_name)

    plt.tight_layout()
    plt.savefig("figs/generated_statistics.png")
    print("Plot saved to figs/generated_statistics.png")


if __name__ == "__main__":
    main()
