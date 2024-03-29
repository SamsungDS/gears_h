import e3x
import flax.linen as nn
import jax.numpy as jnp


class HamiltonianMapper(nn.Module):
    lmax_irrep: int = 4
    lmax_h1: int = 2
    lmax_h2: int = 2

    def setup(self):
        self.cgc = e3x.so3.clebsch_gordan(self.lmax_h1, self.lmax_h2, self.lmax_irrep)

    @nn.compact
    def __call__(self, neighbour_irreps):
        y = e3x.nn.Dense(features=1)(neighbour_irreps)
        # y = neighbour_irreps[..., 0].sum(axis=-3)
        return jnp.einsum("...l,nml->...nm", y[..., 0], self.cgc).sum(axis=-3)
