import flax.linen as nn
import flax.linen
import jax.numpy as jnp

import flax

from slh.layers import (
    SAAtomCenteredDescriptor,
    TDSAAtomCenteredDescriptor,
    BondCenteredTensorMomentDescriptor,
    DenseBlock,
    Readout,
)
from slh.layers.corrections import ExponentialScaleCorrection

from typing import Union


class HamiltonianModel(nn.Module):
    atom_centered: Union[SAAtomCenteredDescriptor,TDSAAtomCenteredDescriptor]
    bond_centered: BondCenteredTensorMomentDescriptor
    dense: DenseBlock
    readout: Readout

    @nn.compact
    def __call__(self, 
                 atomic_numbers, 
                 bc_neighbour_indices, 
                 bc_neighbour_displacements,
                 ac_neighbour_indices, 
                 ac_neighbour_displacements,
                 bond_indices=None):
        atom_centered_descriptors = self.atom_centered(atomic_numbers,
                                                       ac_neighbour_indices, 
                                                       ac_neighbour_displacements
                                                      )

        # atom_centered_descriptors = atom_centered_descriptors.astype(jnp.float32)
        # assert atom_centered_descriptors.dtype == jnp.float32

        bc_features = self.bond_centered(atom_centered_descriptors, 
                                         bc_neighbour_indices.at[bond_indices].get(), 
                                         bc_neighbour_displacements.at[bond_indices].get()
                                        )

        bc_features = bc_features.astype(jnp.float32)

        off_diagonal_denseout = self.dense(bc_features)
        off_diagonal_irreps = self.readout(off_diagonal_denseout)

        on_diagonal_denseout = self.dense(2.0 * atom_centered_descriptors)
        on_diagonal_irreps = Readout(self.readout.nfeatures, self.readout.max_ell)(on_diagonal_denseout)
        # scaling_correction = ExponentialScaleCorrection(
        #     self.readout.nfeatures, self.readout.max_ell
        # )(jnp.linalg.norm(neighbour_displacements, axis=-1, keepdims=True))
        diagonal_scaling = self.param("odscale", flax.linen.initializers.constant(2.0), shape=(1,))
        diagonal_scaling = flax.linen.softplus(diagonal_scaling)
        return off_diagonal_irreps, diagonal_scaling * on_diagonal_irreps  #  * scaling_correction
