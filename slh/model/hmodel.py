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
                cutoff=7.0,
                max_degree=1,
                num_elemental_embedding=32,
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
            cutoff=7.0,
            max_degree=2,
            tensor_module=partial(e3x.nn.FusedTensor, param_dtype=jnp.bfloat16),
        )
    )
    dense: DenseBlock = DenseBlock(
        dense_layer=partial(e3x.nn.Dense, param_dtype=jnp.bfloat16),
        layer_widths=[128, 128],
    )
    readout: Readout = Readout(2, max_ell=2)

    @nn.compact
    def __call__(self, atomic_numbers, neighbour_indices, neighbour_displacements):
        y = self.atom_centered(
            atomic_numbers, neighbour_indices, neighbour_displacements
        )
        # y = jax.vmap(self.bond_centered, in_axes=(None, 0, 0))(y, neighbour_indices, neighbour_displacements)
        y = self.bond_centered(y, neighbour_indices, neighbour_displacements)
        y = self.dense(y)
        h = self.readout(y)
        return h
