import e3x
import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxtyping import Float, Array, Int


e3x.Config.set_cartesian_order(False)


class Readout(nn.Module):
    features: int
    max_ell: int = 4

    @nn.compact
    def __call__(
        self, y: Float[Array, "... num_atompairs 2 (in_ell + 1)**2 in_features"]
    ):
        with jax.ensure_compile_time_eval():
            repeats = 2 * jnp.arange(self.max_ell + 1) + 1
        scaling = self.param(
            "outscale",
            nn.initializers.constant(10.0),
            shape=(2, self.max_ell + 1, self.features),
        )
        return e3x.nn.TensorDense(self.features, self.max_ell, cartesian_order=False)(
            y
        ) * jnp.repeat(scaling, repeats=repeats, axis=-2)
