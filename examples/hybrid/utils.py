import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from abcmodel.atmos.surface_layer.obukhov import ObukhovSurfaceLayerModel


class NeuralNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(1, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 32, rngs=rngs)
        self.linear3 = nnx.Linear(
            32,
            1,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x


class HybridObukhovModel(ObukhovSurfaceLayerModel):
    def __init__(
        self,
        psim_emulator: NeuralNetwork,
        psih_emulator: NeuralNetwork,
    ):
        super().__init__()
        self.psim_emulator = psim_emulator
        self.psih_emulator = psih_emulator

    def compute_psim(self, zeta: Array) -> Array:
        original_shape = zeta.shape
        zeta_safe = jnp.clip(zeta, -5.0, 5.0)
        zeta_input = jax.lax.stop_gradient(zeta_safe)
        res = self.psim_emulator(zeta_input.reshape(-1, 1))
        return res.reshape(original_shape)

    def compute_psih(self, zeta: Array) -> Array:
        original_shape = zeta.shape
        zeta_safe = jnp.clip(zeta, -5.0, 5.0)
        zeta_input = jax.lax.stop_gradient(zeta_safe)
        res = self.psih_emulator(zeta_input.reshape(-1, 1))
        return res.reshape(original_shape)
