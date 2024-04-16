import flax.linen as nn

from slh.layers import (
    AtomCenteredTensorMomentDescriptor,
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis,
    DenseBlock,
    Readout,
)

import e3x
import jax
import jax.numpy as jnp

from functools import partial


class HamiltonianModel(nn.Module):
    atom_centered: AtomCenteredTensorMomentDescriptor = (
        AtomCenteredTensorMomentDescriptor(
            SpeciesAwareRadialBasis(
                cutoff=6.8,
                max_degree=1,
                num_elemental_embedding=64,
                num_radial=64,
                tensor_module=partial(e3x.nn.FusedTensor, param_dtype=jnp.bfloat16),
            ),
            moment_max_degree=2,
            max_moment=2,
            num_moment_features=64,
            use_fused_tensor=True,
        )
    )
    bond_centered: BondCenteredTensorMomentDescriptor = (
        BondCenteredTensorMomentDescriptor(
            cutoff=6.8,
            max_degree=2,
            tensor_module=partial(e3x.nn.FusedTensor, param_dtype=jnp.bfloat16),
        )
    )
    dense: DenseBlock = DenseBlock(
        dense_layer=partial(e3x.nn.Dense, param_dtype=jnp.bfloat16),
        layer_widths=[256, 256],
    )
    readout: Readout = Readout(2, max_ell=2)

    @nn.compact
    def __call__(self, atomic_numbers, neighbour_indices, neighbour_displacements):
        
        atom_centered_descriptors = self.atom_centered(
            atomic_numbers, neighbour_indices, neighbour_displacements
        )

        # atom_centered_descriptors = e3x.nn.MessagePass(max_degree=2)(
        #     inputs=atom_centered_descriptors,
        #     basis=e3x.nn.basis(neighbour_displacements, 1, 32, e3x.nn.smooth_window),
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

        y = self.bond_centered(
            atom_centered_descriptors, neighbour_indices, neighbour_displacements
        )
        
        # y = e3x.nn.soft_sign(y)

        y = self.dense(y)
        h = self.readout(y)
        return h
