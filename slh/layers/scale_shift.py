import e3x
from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array, Int


class OffDiagonalScaleShift(nn.Module):
    # polycoeffs: Float[Array, "num_elements num_elements num_coeffs num_features"]
    exp_prefactors: Float[Array, "num_elements num_elements num_features"]
    exp_lengthscales: Float[Array, "num_elements num_elements num_features"]
    exp_powers: Float[Array, "num_elements num_elements num_features"]

    def __call__(self,
                 x: Float[Array, '... 1 (in_max_degree+1)**2 in_features'],
                 d: Float[Array, ' num_neighbours'],
                 Z_i: Int[Array, ' num_neighbours'], 
                 Z_j: Int[Array, ' num_neighbours']):

        # vmapped_polyval = jax.vmap(jnp.polyval)

        #  polycoeffs = self.polycoeffs[Z_i, Z_j]
        exp_prefactors = self.exp_prefactors[Z_i, Z_j]
        exp_lengthscales = self.exp_lengthscales[Z_i, Z_j]
        exp_powers = self.exp_powers[Z_i, Z_j]

        shifts = exp_prefactors * jnp.exp(- (d / exp_lengthscales) ** exp_powers)



        x = x.at[..., 0, 0, :].add(shifts[..., :])
        # x = e3x.nn.add(self.shifts[atomic_numbers], x)
        return x

class OnDiagonalScaleShift(nn.Module):
    shifts: Float[Array, "num_elements num_features"]
    scales: Float[Array, "num_elements num_features"]

    def __call__(self, 
                 x: Float[Array, '... 1 (in_max_degree+1)**2 in_features'],
                 atomic_numbers: Int[Array, ' num_atoms']):
        x = x.at[..., 0, 0, :].multiply(self.scales[atomic_numbers])
        x = e3x.nn.add(self.shifts[atomic_numbers], x)
        return x
