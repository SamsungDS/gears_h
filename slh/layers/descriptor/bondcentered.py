import e3x
import jax
import jax.numpy as jnp

from functools import partial
import flax.linen as nn

from slh.layers.descriptor.radial_basis import jinclike

from typing import Optional, Union


class BondCenteredTensorMomentDescriptor(nn.Module):
    cutoff: float
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
        # y = self.tensor_module(
        #     max_degree=self.max_actp_degree,
        #     name="atompair_tp",
        #     cartesian_order=False,
        #     dtype=jnp.float32,
        # )(atom1_desc, atom2_desc)

        y = e3x.nn.add(atom1_desc, atom2_desc)

        # We put in information about the orientation/length of the bond vector here
        bond_expansion = e3x.nn.basis(
            neighbour_displacements,
            num=num_radial_features,
            max_degree=self.max_basis_degree,
            radial_fn=partial(jinclike, limit=self.cutoff),
            cartesian_order=False,
        ).astype(jnp.float32)
        bond_expanded_dense = bond_expansion # e3x.nn.Dense(num_radial_features)(bond_expansion)

        # num_pairs x 2 x (max_degree + 1)^2 x num_radial_features
        # y = e3x.nn.add(y, bond_expansion)
        tp = self.tensor_module(
            max_degree=self.max_degree, cartesian_order=False, dtype=jnp.float32
        )
        y = tp(bond_expanded_dense, y)

        return y
