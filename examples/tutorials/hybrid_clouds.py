import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import Array

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.atmos.clouds.cumulus import CumulusModel
from abcmodel.integration import outter_step
from abcmodel.utils import create_dataloader, get_path_string


class NeuralNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 32, rngs=rngs)
        self.linear3 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x


class HybridCumulusModel(CumulusModel):
    def __init__(self, net: NeuralNetwork):
        # the original CumulusModel does not have any parameters,
        # but otherwise we would need to pass them to init the parent class
        super().__init__()
        self.net = net
        # this means/stds will be modified one we know the exact values for normalization
        # which we will get once we separate the dataset into train and test
        self.x_in_mean = nnx.BatchStat(jnp.array([0.0, 0.0, 0.0, 0.0]))
        self.x_in_std = nnx.BatchStat(jnp.array([1.0, 1.0, 1.0, 1.0]))
        self.x_out_mean = nnx.BatchStat(jnp.array(0.0))
        self.x_out_std = nnx.BatchStat(jnp.array(1.0))

    def compute_cc_frac(
        self,
        q: Array,
        top_T: Array,
        top_p: Array,
        q2_h: Array,
    ) -> Array:
        # this is effectively our only replacement from the original class
        # instead of inheriting this method, we are now using the neural network
        # defined above to calculate cc_frac as a function of the same variables
        # that the original function used: q, top_T, top_p and q2_h
        x = jnp.array([q, top_T, top_p, q2_h])
        # note that, before using the neural network, we normalize these variables
        x = (x - self.x_in_mean.value) / self.x_in_std.value
        # we squeeze the output to maintain the same shape through the ABC-Model
        x = jnp.squeeze(self.net(x))
        # and we re-normalize the output too
        x = x * self.x_out_std.value + self.x_out_mean.value
        return x


def load_model_and_template_state(key: Array):
    # here we are going to build the model in the standard way like we always do,
    # but the cloud model will now be replaced with out hybrid version!

    # radiation
    rad_model_kwargs = cm.standard_radiation.model_kwargs
    rad_model = abcmodel.rad.StandardRadiationModel(**rad_model_kwargs)
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # land
    ags_model_kwargs = cm.ags.model_kwargs
    land_model = abcmodel.land.AgsModel(**ags_model_kwargs)

    ags_state_kwargs = cm.ags.state_kwargs
    land_state = land_model.init_state(**ags_state_kwargs)

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer
    mixed_state_kwargs = cm.bulk_mixed_layer.state_kwargs
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs
    )
    mixed_layer_state = mixed_layer_model.init_state(**mixed_state_kwargs)

    # clouds
    net = NeuralNetwork(rngs=nnx.Rngs(key))
    cloud_model = HybridCumulusModel(net)
    cloud_state = cloud_model.init_state()

    # atmosphere
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_state, mixed=mixed_layer_state, clouds=cloud_state
    )

    # coupler
    abcoupler = abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    return abcoupler, state


def load_batched_data(key: Array, template_state, ratio: float = 0.8):
    # here we read the dataset that we generated using
    # the perturbed initial conditions in data/generate.py
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
    # 1) x = state[t], y = target_var[t+1]
    # 2) shapes: (num_ens, num_times) -> (num_ens * num_times)
    def prep_input(arr):
        sliced = arr[:, :-1]
        return sliced.reshape(-1)

    def prep_target(arr):
        sliced = arr[:, 1:]
        return sliced.reshape(-1)

    # apply prep functions
    # we need to use jax.tree.map for x, because it is not an array,
    # but the CoupledState of the model: a PyTree with arrays in the leaves
    # the jax.tree.map function wil then apply the prep_input function to all
    # the leaves (variables within each component of the model)
    x_full = jax.tree.map(prep_input, traj_ensembles)
    y_full = prep_target(traj_ensembles.atmos.clouds.cc_frac)

    # train/test split
    num_samples = y_full.shape[0]
    split_idx = int(ratio * num_samples)
    idxs = jax.random.permutation(key, num_samples)
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]

    # helper to slice a PyTree by index once again using
    # jax.tree.map to reach the leaves of the PyTree
    def subset(tree, indices):
        return jax.tree.map(lambda x: x[indices], tree)

    x_train = subset(x_full, train_idxs)
    x_test = subset(x_full, test_idxs)
    y_train = y_full[train_idxs]
    y_test = y_full[test_idxs]

    return x_train, x_test, y_train, y_test


def get_norms(x, y):
    # this is pretty standard, but we don't take the norm of the entire state
    # only of the variables that we will need for the neural net
    x_in_mean = jnp.array(
        [
            jnp.mean(x.atmos.mixed.q),
            jnp.mean(x.atmos.mixed.top_T),
            jnp.mean(x.atmos.mixed.top_p),
            jnp.mean(x.atmos.clouds.q2_h),
        ]
    )
    x_in_std = jnp.array(
        [
            jnp.std(x.atmos.mixed.q),
            jnp.std(x.atmos.mixed.top_T),
            jnp.std(x.atmos.mixed.top_p),
            jnp.std(x.atmos.clouds.q2_h),
        ]
    )
    x_out_mean = jnp.mean(x.atmos.clouds.cc_frac)
    x_out_std = jnp.std(x.atmos.clouds.cc_frac)
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)

    return x_in_mean, x_in_std, x_out_mean, x_out_std, y_mean, y_std


def train(
    model,
    template_state,
    inner_dt: float,
    outter_dt: float,
    tstart: float,
    lr: float = 1e-5,
    batch_size: int = 4,
    epochs: int = 3,
    print_every: int = 100,
):
    inner_tsteps = int(outter_dt / inner_dt)

    # data setup
    key = jax.random.PRNGKey(42)
    data_key, train_key = jax.random.split(key)
    x_train, x_test, y_train, y_test = load_batched_data(data_key, template_state)
    x_in_mean, x_in_std, x_out_mean, x_out_std, y_mean, y_std = get_norms(
        x_train, y_train
    )
    # we re-populate the dummy array we allocated in the neural net above
    # with the true stats for the mean and standard deviation
    # this is a hacky trick :P (and maybe there is a cleaner way to do this)
    model.atmos.clouds.x_in_mean.value = x_in_mean
    model.atmos.clouds.x_in_std.value = x_in_std
    model.atmos.clouds.x_out_mean.value = x_out_mean
    model.atmos.clouds.x_out_std.value = x_out_std

    # optimizer
    optimizer = nnx.Optimizer(
        # this is the entire model with the mechanistic and neural network parts
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.radam(lr)),
        # but the optimizer needs takes gradient only w.r.t. nnx.Params
        wrt=nnx.Param,
    )

    def loss_fn(model, x_batch_state, y_batch):
        def run_single(state):
            # limamau: are we supposed to use the last step of our avg traj here?
            final_state, avg_traj = outter_step(
                state,
                None,
                coupler=model,
                inner_dt=inner_dt,
                inner_tsteps=inner_tsteps,
                tstart=tstart,
            )
            return avg_traj

        # here jax.vmap is applying the run_single function
        # over each one of the dimensions of x_batch_state
        # over axis=0, which is the batch axis, so we are
        # essentially parallelizing the model over the batches
        pred_state = jax.vmap(run_single)(x_batch_state)
        # we select only what chose to be our "observation"
        pred_le = pred_state.atmos.clouds.cc_frac
        # quick normalization for the loss
        pred_le_norm = (pred_le - y_mean) / y_std
        y_batch_norm = (y_batch - y_mean) / y_std
        # and this is the standard mean squared error (MSE)
        return jnp.mean((pred_le_norm - y_batch_norm) ** 2)

    # it is important to jit the update_step during training loops
    @nnx.jit
    def update_step(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)

        # replace any NaN in the gradients with 0.0
        grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)

        # !!!! important !!!!
        # for old jax versions (e.g., 0.4.38)
        # take out the model fmor the update
        optimizer.update(model, grads)

        return loss

    print(f"training on {y_train.shape[0]} samples;")
    print(f"training for {epochs} epochs with batch size {batch_size};")
    print(f"printing avg loss every {print_every} steps...")

    # training loop
    total_loss = 0.0
    step = 0
    last_printed_loss = jnp.inf
    is_early_stopping = False
    for _ in range(epochs):
        train_key, subkey = jax.random.split(train_key)
        loader = create_dataloader(x_train, y_train, batch_size, subkey)
        for x_batch, y_batch in loader:
            loss = update_step(model, optimizer, x_batch, y_batch)
            total_loss += loss
            if (step % print_every == 0) & (step > 0):
                avg_loss = total_loss / print_every
                print(f"step {step} | loss: {avg_loss:.6f}", flush=True)
                total_loss = 0.0
                if avg_loss < last_printed_loss:
                    last_printed_loss = avg_loss
                else:
                    is_early_stopping = True
                    break
            step += 1
        if is_early_stopping:
            break

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
