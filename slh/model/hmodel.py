import flax.linen as nn
import jax.numpy as jnp

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
    def __call__(self, atomic_numbers, neighbour_indices, neighbour_displacements):
        atom_centered_descriptors = self.atom_centered(
            atomic_numbers, neighbour_indices, neighbour_displacements
        )

        atom_centered_descriptors = atom_centered_descriptors.astype(jnp.float32)
        assert atom_centered_descriptors.dtype == jnp.float32

        bc_features = self.bond_centered(
            atom_centered_descriptors, neighbour_indices, neighbour_displacements
        )

        bc_features = bc_features.astype(jnp.float32)

        off_diagonal_denseout = self.dense(bc_features)
        off_diagonal_irreps = self.readout(off_diagonal_denseout)
        # scaling_correction = ExponentialScaleCorrection(
        #     self.readout.nfeatures, self.readout.max_ell
        # )(jnp.linalg.norm(neighbour_displacements, axis=-1, keepdims=True))
        return off_diagonal_irreps  #  * scaling_correction
