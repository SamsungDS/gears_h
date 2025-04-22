import flax.linen as nn
import jax.numpy as jnp

from slh.layers import (
    SAAtomCenteredDescriptor,
    TDSAAtomCenteredDescriptor,
    BondCenteredTensorMomentDescriptor,
    DenseBlock,
    Readout,
    OffDiagonalScaleShift,
    OnDiagonalScaleShift
)
from slh.layers import LayerNorm

from typing import Union


class HamiltonianModel(nn.Module):
    atom_centered: Union[SAAtomCenteredDescriptor,TDSAAtomCenteredDescriptor]
    bond_centered: BondCenteredTensorMomentDescriptor
    dense: DenseBlock
    off_diag_readout: Readout
    on_diag_readout: Readout
    off_diag_scale_shift: OffDiagonalScaleShift
    on_diag_scale_shift: OnDiagonalScaleShift

    @nn.compact
    def __call__(self, 
                 atomic_numbers, 
                 bc_neighbour_indices, 
                 bc_neighbour_displacements,
                 ac_neighbour_indices, 
                 ac_neighbour_displacements
                 ):
        atom_centered_descriptors = self.atom_centered(atomic_numbers,
                                                       ac_neighbour_indices, 
                                                       ac_neighbour_displacements
                                                      )

        bc_features = self.bond_centered(atom_centered_descriptors, 
                                         bc_neighbour_indices, 
                                         bc_neighbour_displacements
                                        )

        bc_features = bc_features.astype(jnp.float32)

        off_diagonal_denseout = self.dense(bc_features)
        off_diagonal_denseout = LayerNorm()(off_diagonal_denseout)
        
        off_diagonal_irreps = self.off_diag_readout(off_diagonal_denseout)
        scaled_off_diagonal_irreps = self.off_diag_scale_shift(off_diagonal_irreps,
                                                               jnp.linalg.norm(bc_neighbour_displacements, axis=1),
                                                               atomic_numbers[bc_neighbour_indices[:,0]],
                                                               atomic_numbers[bc_neighbour_indices[:,1]]
                                                               )

        on_diagonal_denseout = self.dense(atom_centered_descriptors)
        on_diagonal_denseout = LayerNorm()(on_diagonal_denseout)
        
        on_diagonal_irreps = self.on_diag_readout(on_diagonal_denseout)
        scaled_on_diagonal_irreps = self.on_diag_scale_shift(on_diagonal_irreps,
                                                             atomic_numbers)
        
        return scaled_off_diagonal_irreps, scaled_on_diagonal_irreps
