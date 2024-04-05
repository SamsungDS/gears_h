import flax.linen as nn

from surrogatelcaohamiltonians.layers import (
    AtomCenteredTensorMomentDescriptor,
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis,
    DenseBlock,
    Readout,
)


class HamiltonianModel(nn.Module):
    atom_centered: AtomCenteredTensorMomentDescriptor = AtomCenteredTensorMomentDescriptor(
        SpeciesAwareRadialBasis(cutoff=8.0),
    )
    bond_centered: BondCenteredTensorMomentDescriptor = BondCenteredTensorMomentDescriptor(cutoff=8.0)
    dense: DenseBlock = DenseBlock()
    readout: Readout = Readout(2, max_ell=2)

    @nn.compact
    def __call__(self, atomic_numbers, neighbour_indices, neighbour_displacements):
        y = self.atom_centered(
            atomic_numbers, neighbour_indices, neighbour_displacements
        )
        y = self.bond_centered(y, neighbour_indices, neighbour_displacements)
        y = self.dense(y)
        h = self.readout(y)
        return h
