import e3x
import jax
import jax.numpy as jnp

from functools import partial
import functools
import flax.linen as nn

from typing import Optional, Union


class BondCenteredTensorMomentDescriptor(nn.Module):
    cutoff: float
    # num_basis_features: int = 8
    # basis_max_degree: int = 3
    # num_moment_features: int = 64 # TODO this can in principle be a list of ints
    max_degree: int = 4
    tensor_module: Union[e3x.nn.Tensor, e3x.nn.FusedTensor] = e3x.nn.FusedTensor
    # radial_function = e3x.nn.basic_fourier

    @nn.compact
    def __call__(self, atomic_descriptors, neighbour_indices, neighbour_displacements):
        neighbours_i, neighbours_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        num_radial_features = atomic_descriptors.shape[-1]

        atom1_desc, atom2_desc = (
            atomic_descriptors[neighbours_i],
            atomic_descriptors[neighbours_j],
        )
        y = self.tensor_module(max_degree=self.max_degree, name="atompair_td", cartesian_order=False)(atom1_desc, atom2_desc)

        # We put in information about the orientation/length of the bond vector here
        bond_expansion = e3x.nn.basis(
            neighbour_displacements,
            num=num_radial_features,
            max_degree=self.max_degree,
            radial_fn=partial(e3x.nn.basic_gaussian, limit=self.cutoff),
            cutoff_fn=partial(e3x.nn.cosine_cutoff, cutoff=self.cutoff),
            cartesian_order=False
        )

        # num_pairs x 2 x (max_degree + 1)^2 x num_radial_features
        y = e3x.nn.add(
            y, bond_expansion
        )  
        # y = self.tensor_module(max_degree=self.max_degree)(y, bond_expansion)

        # TODO, in principle, we can put in the element-pair information here as a
        # residual connection, but I can't think why we would need to.

        return y
