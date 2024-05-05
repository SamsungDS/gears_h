import e3x
import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxtyping import Float, Array, Int


class Readout(nn.Module):
    nfeatures: int
    max_ell: int = 4

    @nn.compact
    def __call__(
        self, y: Float[Array, "... num_atompairs 2 (in_ell + 1)**2 in_features"]
    ):
        return e3x.nn.TensorDense(
            self.nfeatures, self.max_ell, cartesian_order=False, dtype=jnp.float32
        )(y)
