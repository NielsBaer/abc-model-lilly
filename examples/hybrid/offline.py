import os

import h5py
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import nnx
from jax import Array
from utils import HybridObukhovModel, NeuralNetwork

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.integration import outter_step
from abcmodel.utils import get_path_string

# to maybe debug NaNs
jax.config.update("jax_debug_nans", True)


def load_model_and_template_state(key: Array):
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

    return abcoupler, state


def load_batched_data(key, template_state, ratio=0.8):
    """Loads data into the State structure."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../data/dataset.h5")

    def load_leaf(path, leaf_template):
        path_str = get_path_string(path)
        with h5py.File(file_path, "r") as f:
            data = jnp.array(f[path_str])
            return data

    print("loading data structure...")
    # This magic function walks the template state and loads the matching H5 data for every variable
    full_history = jtu.tree_map_with_path(load_leaf, template_state)

    # --- C. Time Shifting & Shaping ---
    # We want: Input = State[t], Target = LE[t+1]

    # 1. Flatten Ensembles and Time into one "Batch" dimension
    # Current leaves: (N_ensembles, Time_steps)
    # We slice first, then flatten.

    def prep_input(arr):
        # Take all times except the last one (0 to T-1)
        sliced = arr[:, :-1]
        # Flatten (N, T-1) -> (N * (T-1))
        return sliced.reshape(-1)

    def prep_target(arr):
        # Take all times starting from 1 (1 to T)
        sliced = arr[:, 1:]
        return sliced.reshape(-1)

    # Apply to the whole state tree to get x (Input State)
    x_full = jtu.tree_map(prep_input, full_history)

    # Extract just LE for y (Target)
    y_full = prep_target(full_history.land.le)

    # --- D. Train/Test Split ---
    num_samples = y_full.shape[0]
    split_idx = int(ratio * num_samples)

    # Shuffle indices
    idxs = jax.random.permutation(key, num_samples)
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]

    # Helper to slice a PyTree by index
    def subset(tree, indices):
        return jtu.tree_map(lambda x: x[indices], tree)

    x_train = subset(x_full, train_idxs)
    x_test = subset(x_full, test_idxs)
    y_train = y_full[train_idxs]
    y_test = y_full[test_idxs]

    return x_train, x_test, y_train, y_test


def normalize_tree(tree, mean_tree, std_tree):
    return jtu.tree_map(lambda x, m, s: (x - m) / s, tree, mean_tree, std_tree)


def unnormalize_tree(tree, mean_tree, std_tree):
    return jtu.tree_map(lambda x, m, s: x * s + m, tree, mean_tree, std_tree)


def create_dataloader(x_state, y, batch_size, key):
    """Yields batches: x_state is a PyTree, y is an array."""
    num_samples = y.shape[0]
    indices = jax.random.permutation(key, num_samples)
    num_batches = num_samples // batch_size

    def get_batch(tree, idxs):
        return jtu.tree_map(lambda x: x[idxs], tree)

    for i in range(num_batches):
        batch_idx = indices[i * batch_size : (i + 1) * batch_size]
        yield get_batch(x_state, batch_idx), y[batch_idx]


def train(model, template_state):
    # config
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    tstart = 6.5
    inner_tsteps = int(outter_dt / inner_dt)
    lr = 1e-5
    batch_size = 4
    epochs = 100

    # data setup
    key = jax.random.PRNGKey(42)
    data_key, train_key = jax.random.split(key)
    x_train, x_test, y_train, y_test = load_batched_data(data_key, template_state)

    y_mean, y_std = jnp.mean(y_train), jnp.std(y_train)
    print(f"training on {y_train.shape[0]} samples...")

    # optimizer
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)),
        wrt=nnx.Param,
    )

    def loss_fn(model, x_batch_state, y_batch):
        # here, x_batch_state is a CoupledState object with physical values
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
        grad_norm = optax.global_norm(grads)
        jax.debug.print("Loss: {}, Grad Norm: {}", loss, grad_norm)
        optimizer.update(grads)
        return loss

    # training loop
    for epoch in range(epochs):
        train_key, subkey = jax.random.split(train_key)
        loader = create_dataloader(x_train, y_train, batch_size, subkey)

        total_loss = 0.0
        count = 0

        for x_batch, y_batch in loader:
            loss = update_step(model, optimizer, x_batch, y_batch)
            print("loss:", loss)
            total_loss += loss
            count += 1

        print(f"epoch {epoch + 1} | loss: {total_loss / count:.6f}")

    return model


def main():
    key = jax.random.PRNGKey(42)
    model, template_state = load_model_and_template_state(key)
    train(model, template_state)


if __name__ == "__main__":
    main()
