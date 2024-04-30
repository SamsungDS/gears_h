import e3x
import jax
import jax.numpy as jnp

from functools import partial
import flax.linen as nn

from slh.layers.descriptor.radial_basis import jinclike

from typing import Optional, Union


class BondCenteredTensorMomentDescriptor(nn.Module):
    cutoff: float
    # num_basis_features: int = 8
    # basis_max_degree: int = 3
    # num_moment_features: int = 64 # TODO this can in principle be a list of ints
    max_actp_degree: int = 4
    max_basis_degree: int = 2
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
        y = self.tensor_module(
            max_degree=self.max_actp_degree,
            name="atompair_tp", cartesian_order=False
        )(atom1_desc, atom2_desc)

        # We put in information about the orientation/length of the bond vector here
        bond_expansion = e3x.nn.basis(
            neighbour_displacements,
            num=num_radial_features,
            max_degree=self.max_basis_degree,
            radial_fn=partial(jinclike, limit=self.cutoff),
            cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
            damping_fn=partial(e3x.nn.smooth_damping, gamma=5.0),
            cartesian_order=False,
        )

        # num_pairs x 2 x (max_degree + 1)^2 x num_radial_features
        # y = e3x.nn.add(y, bond_expansion)
        y = self.tensor_module(max_degree=self.max_degree, cartesian_order=False, 
                               # kernel_init=partial(e3x.nn.initializers.fused_tensor_normal, scale=1e-2)
                               )(
            y, bond_expansion
        )
        # y = jnp.concat([y, e3x.nn.change_max_degree_or_type(bond_expansion, include_pseudotensors=True)], axis=-1)

        return y
