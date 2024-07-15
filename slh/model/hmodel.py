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

        # atom_centered_descriptors = atom_centered_descriptors.astype(jnp.float32)
        # assert atom_centered_descriptors.dtype == jnp.float32

        # atom_centered_descriptors = e3x.nn.MessagePass(max_degree=2, cartesian_order=False)(
        #     inputs=atom_centered_descriptors,
        #     basis=e3x.nn.basis(neighbour_displacements, max_degree=1, num=8, radial_fn=e3x.nn.smooth_window, cartesian_order=False),
        #     src_idx=neighbour_indices[:, 1],
        #     dst_idx=neighbour_indices[:, 0],
        #     num_segments=len(atom_centered_descriptors))
        # atom_centered_descriptors = atom_centered_descriptors / 45.0

        # atom_centered_descriptors = e3x.nn.soft_sign(atom_centered_descriptors)

        # atom_centered_descriptors = e3x.nn.MessagePass(max_degree=2)(
        #     inputs=atom_centered_descriptors,
        #     basis=e3x.nn.basis(neighbour_displacements, 1, 32, e3x.nn.smooth_window),
        #     src_idx=neighbour_indices[:, 1],
        #     dst_idx=neighbour_indices[:, 0],
        #     num_segments=len(atom_centered_descriptors))
        # y = jax.vmap(self.bond_centered, in_axes=(None, 0, 0))(y, neighbour_indices, neighbour_displacements)

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
