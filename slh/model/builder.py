from slh.layers.descriptor import (
    AtomCenteredTensorMomentDescriptor,
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis,
)

from slh.layers.residual_dense import DenseBlock
from slh.layers.readout import Readout


def build_model(config, readout_config):
    radial_descriptor = SpeciesAwareRadialBasis(
        cutoff=7.0, num_radial=32, max_degree=2, name="radial basis"
    )

    atom_centered_descriptor = AtomCenteredTensorMomentDescriptor(
        radial_basis=radial_descriptor, name="atomcentered descriptor"
    )

    bond_descriptor = BondCenteredTensorMomentDescriptor(
        cutoff=7.0, name="bond descriptor"
    )

    neural_net = DenseBlock()

    readout = Readout(
        features=5,
    )
