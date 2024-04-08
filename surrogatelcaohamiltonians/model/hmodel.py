import flax.linen as nn

from surrogatelcaohamiltonians.layers import (
    AtomCenteredTensorMomentDescriptor,
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis,
    DenseBlock,
    Readout,
)

import e3x
import jax


class HamiltonianModel(nn.Module):
    atom_centered: AtomCenteredTensorMomentDescriptor = (
        AtomCenteredTensorMomentDescriptor(
            SpeciesAwareRadialBasis(
                cutoff=6.0,
                max_degree=2,
                num_elemental_embedding=64,
                num_radial=64,
                tensor_module=e3x.nn.FusedTensor,
            ),
            moment_max_degree=4,
            num_moment_features=64,
            use_fused_tensor=True,
        )
    )
    bond_centered: BondCenteredTensorMomentDescriptor = (
        BondCenteredTensorMomentDescriptor(
            cutoff=6.0, max_degree=4, tensor_module=e3x.nn.FusedTensor
        )
    )
    dense: DenseBlock = DenseBlock(layer_width=128)
    readout: Readout = Readout(4, max_ell=4)

    @nn.compact
    def __call__(self, atomic_numbers, neighbour_indices, neighbour_displacements):
        y = self.atom_centered(
            atomic_numbers, neighbour_indices, neighbour_displacements
        )
        y = self.bond_centered(y, neighbour_indices, neighbour_displacements)
        y = self.dense(y)
        h = self.readout(y)
        return h
