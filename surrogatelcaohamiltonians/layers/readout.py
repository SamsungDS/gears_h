import e3x
import jax
import flax.linen as nn

from jaxtyping import Float, Array, Int

from surrogatelcaohamiltonians.layers.descriptor.radial_basis import (
    SpeciesAwareRadialBasis,
)

e3x.Config.set_cartesian_order(False)


class Readout(nn.Module):
    features: int
    max_ell: int = 4

    @nn.compact
    def __call__(self, y):
        return e3x.nn.TensorDense(self.features, max_degree=self.max_ell)(y)