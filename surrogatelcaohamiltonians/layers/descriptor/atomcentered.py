
import e3x
import jax
import jax.numpy as jnp

from functools import partial
import functools
import flax.linen as nn

from typing import Optional

e3x.Config.set_cartesian_order(False)


class AtomCenteredTensorMomentDescriptor(nn.Module):
  cutoff: float
  num_basis_features: int = 8
  basis_max_degree: int = 3
  num_moment_features: int = 64 # TODO this can in principle be a list of ints
  moment_max_degree: int = 4
  max_moment: int = 2
  tensor_module: nn.Module = e3x.nn.TensorDense

  def setup(self):
    self.radial_function = partial(e3x.nn.sinc, limit=self.cutoff)

  @nn.compact
  def __call__(self, neighbour_displacements):

    y = e3x.nn.basis(neighbour_displacements,
                     max_degree=self.basis_max_degree,
                     num=self.num_basis_features,
                     radial_fn=self.radial_function,
                     cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
                    )
    # The line above is our 'first moment' since we have directional information.
    # Each subsequent moment happens down here.
    for _ in range(self.max_moment - 1):
      y = self.tensor_module(features=self.num_moment_features, max_degree=self.moment_max_degree)(y)
    
    return y