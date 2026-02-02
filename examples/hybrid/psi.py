import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import Array
from utils import HybridObukhovModel, NeuralNetwork

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.integration import outter_step
from abcmodel.utils import get_path_string


def load_model_and_template_state(key: Array):
    """Loads the standard model and it's template state."""
    psim_key, psih_key = jax.random.split(key)
    psim_net = NeuralNetwork(rngs=nnx.Rngs(psim_key))
    psih_net = NeuralNetwork(rngs=nnx.Rngs(psih_key))

    # radiation
    rad_model_kwargs = cm.standard_radiation.model_kwargs
    rad_model = abcmodel.rad.StandardRadiationModel(**rad_model_kwargs)
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # land
    ags_model_kwargs = cm.ags.model_kwargs
    land_model = abcmodel.land.AgsModel(**ags_model_kwargs)

    ags_state_kwargs = cm.ags.state_kwargs
    land_state = land_model.init_state(**ags_state_kwargs)

    # surface layer (the one we build with the neural nets!)
    surface_layer_model = HybridObukhovModel(psim_net, psih_net)
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer
    mixed_state_kwargs = cm.bulk_mixed_layer.state_kwargs
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
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

    return abcoupler, state


def load_batched_data(key: Array, template_state, ratio: float = 0.8):
    """Loads data into the State structure."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../data/dataset.h5")

    def load_leaf(path, leaf_template):
        path_str = get_path_string(path)
        with h5py.File(file_path, "r") as f:
            data = jnp.array(f[path_str])
            return data

    print("loading data structure...")
    # here we walk through the template state and
    # load the matching .h5 data for every variable
    traj_ensembles = jax.tree.map_with_path(load_leaf, template_state)

    # the two prep functions follow:
    # 1) x = state[t], y = LE[t+1]
    # 2) shapes: (num_ens, num_times) -> (num_ens * num_times)
    def prep_input(arr):
        # take all times except the last one (0 to T-1)
        sliced = arr[:, :-1]
        # Flatten (N, T-1) -> (N * (T-1))
        return sliced.reshape(-1)

    def prep_target(arr):
        # take all times starting from 1 (1 to T)
        sliced = arr[:, 1:]
        return sliced.reshape(-1)

    # apply prep functions
    x_full = jax.tree.map(prep_input, traj_ensembles)
    # our target is latent heat
    y_full = prep_target(traj_ensembles.land.le)

    # train/test split
    num_samples = y_full.shape[0]
    split_idx = int(ratio * num_samples)

    # shuffle indices
    idxs = jax.random.permutation(key, num_samples)
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]

    # helper to slice a PyTree by index
    def subset(tree, indices):
        return jax.tree.map(lambda x: x[indices], tree)

    x_train = subset(x_full, train_idxs)
    x_test = subset(x_full, test_idxs)
    y_train = y_full[train_idxs]
    y_test = y_full[test_idxs]

    return x_train, x_test, y_train, y_test


def normalize_tree(tree, mean_tree: Array, std_tree: Array):
    return jax.tree.map(lambda x, m, s: (x - m) / s, tree, mean_tree, std_tree)


def unnormalize_tree(tree, mean_tree: Array, std_tree: Array):
    return jax.tree.map(lambda x, m, s: x * s + m, tree, mean_tree, std_tree)


def create_dataloader(x_state, y: Array, batch_size: int, key: Array):
    """Yields batches: x_state is a PyTree, y is an array."""
    num_samples = y.shape[0]
    indices = jax.random.permutation(key, num_samples)
    num_batches = num_samples // batch_size

    def get_batch(tree, idxs):
        return jax.tree.map(lambda x: x[idxs], tree)

    for i in range(num_batches):
        batch_idx = indices[i * batch_size : (i + 1) * batch_size]
        yield get_batch(x_state, batch_idx), y[batch_idx]


def train(
    model,
    template_state,
    inner_dt: float,
    outter_dt: float,
    tstart: float,
    lr: float = 1e-5,
    batch_size: int = 4,
    epochs: int = 1,
    print_every: int = 100,
):
    # config
    inner_tsteps = int(outter_dt / inner_dt)

    # data setup
    key = jax.random.PRNGKey(42)
    data_key, train_key = jax.random.split(key)
    x_train, x_test, y_train, y_test = load_batched_data(data_key, template_state)
    y_mean, y_std = jnp.mean(y_train), jnp.std(y_train)

    # optimizer
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)),
        wrt=nnx.Param,
    )

    def loss_fn(model, x_batch_state, y_batch):
        def run_single(state):
            final_state, _ = outter_step(
                state,
                None,
                coupler=model,
                inner_dt=inner_dt,
                inner_tsteps=inner_tsteps,
                tstart=tstart,
            )
            return final_state

        pred_state = jax.vmap(run_single)(x_batch_state)
        pred_le = pred_state.land.le
        pred_le_norm = (pred_le - y_mean) / y_std
        y_batch_norm = (y_batch - y_mean) / y_std
        return jnp.mean((pred_le_norm - y_batch_norm) ** 2)

    @nnx.jit
    def update_step(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)

        # replace any NaN in the gradients with 0.0
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)

        # clip gradients to prevent "real" explosions
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        optimizer.update(grads)

        return loss

    print(f"training on {y_train.shape[0]} samples")
    print(f"training for {epochs} epochs with batch size {batch_size}")
    print(f"printing avg loss every {print_every} steps...")

    # training loop
    total_loss = 0.0
    step = 0
    for _ in range(epochs):
        train_key, subkey = jax.random.split(train_key)
        loader = create_dataloader(x_train, y_train, batch_size, subkey)
        for x_batch, y_batch in loader:
            loss = update_step(model, optimizer, x_batch, y_batch)
            total_loss += loss
            if step % print_every == 0:
                print(f"step {step} | loss: {total_loss / print_every:.6f}")
                total_loss = 0.0
            step += 1

    return model


def benchmark_plot(
    hybrid_coupler,
    inner_dt: float,
    outter_dt: float,
    runtime: float,
    tstart: float,
):
    # rad with clouds
    rad_model = abcmodel.rad.CloudyRadiationModel(
        **cm.cloudy_radiation.model_kwargs,
    )
    rad_state = rad_model.init_state(**cm.cloudy_radiation.state_kwargs)

    # land surface
    land_model = abcmodel.land.AgsModel(
        **cm.ags.model_kwargs,
    )
    land_state = land_model.init_state(
        **cm.ags.state_kwargs,
    )

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    mixed_layer_state = mixed_layer_model.init_state(
        **cm.bulk_mixed_layer.state_kwargs,
    )

    # clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # define atmos model
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

    # define coupler and coupled state
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

    # run run run
    time, std_trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )
    time, hybrid_trajectory = abcmodel.integrate(
        state, hybrid_coupler, inner_dt, outter_dt, runtime, tstart
    )

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, std_trajectory.atmos.mixed.h_abl)
    plt.plot(time, hybrid_trajectory.atmos.mixed.h_abl)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, std_trajectory.atmos.mixed.theta)
    plt.plot(time, hybrid_trajectory.atmos.mixed.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, std_trajectory.atmos.mixed.q * 1000.0)
    plt.plot(time, hybrid_trajectory.atmos.mixed.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, std_trajectory.atmos.clouds.cc_frac)
    plt.plot(time, hybrid_trajectory.atmos.clouds.cc_frac)
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, std_trajectory.land.gf, label="standard")
    plt.plot(time, hybrid_trajectory.land.gf, label="hybrid")
    plt.xlabel("time [h]")
    plt.ylabel("ground heat flux [W m-2]")
    plt.legend()

    plt.subplot(236)
    plt.plot(time, std_trajectory.land.le_veg)
    plt.plot(time, hybrid_trajectory.land.le_veg)
    plt.xlabel("time [h]")
    plt.ylabel("latent heat flux from vegetation [W m-2]")

    plt.tight_layout()
    plt.show()


def main():
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    runtime = 12 * 3600.0
    tstart = 6.5
    key = jax.random.PRNGKey(42)
    hybrid_model, template_state = load_model_and_template_state(key)
    hybrid_model = train(hybrid_model, template_state, inner_dt, outter_dt, tstart)
    benchmark_plot(
        hybrid_model,
        inner_dt=inner_dt,
        outter_dt=outter_dt,
        runtime=runtime,
        tstart=tstart,
    )


if __name__ == "__main__":
    main()
