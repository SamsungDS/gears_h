
import e3x
import jax
import jax.numpy as jnp

from functools import partial
import functools
import flax.linen as nn

from typing import Optional

e3x.Config.set_cartesian_order(False)


class AtomCenteredTensorMomentDescriptor(nn.Module):
  radial_function: e3x.nn.sinc
  cutoff: float
  basis_max_degree: int = 3
  moment_max_degree: int = 4
  max_moment: 3

  def setup(self):
    pass

  def __call__(self, atomic_numbers, neighbour_displacements, neighbour_indices):

    y = e3x.nn.basis(neighbour_displacements,
                                      max_degree=self.basis_max_degree,
                                      cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
                                      )
    for moment in range(self.max_moment - 1):
      y = e3x.nn.FusedTensor(max_degree=self.moment_max_degree)(y)
    
    return y
