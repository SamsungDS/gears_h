from functools import partial

import e3x
import jax.numpy as jnp
import numpy as np

import slh
from slh.config.train_config import ModelConfig
from slh.layers.descriptor import (
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis
)
from slh.layers.readout import Readout
from slh.layers.residual_dense import DenseBlock
from slh.layers.scale_shift import OffDiagonalScaleShift, OnDiagonalScaleShift
from slh.model.hmodel import HamiltonianModel

class ModelBuilder:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def build_species_aware_radial_basis(self):
        radial_config = self.config['atom_centered']['radial_basis']
        
        sarb = SpeciesAwareRadialBasis(cutoff=radial_config["cutoff"],
                                       num_radial=radial_config["num_radial"],
                                       max_degree=radial_config["max_degree"],
                                       num_elemental_embedding=radial_config["num_elemental_embedding"]
                                      )
        return sarb
    
    def build_atom_centered_descriptor(self):
        radial_basis = self.build_species_aware_radial_basis()

        ac_config = self.config['atom_centered']
        descriptor_name = ac_config['descriptor']['descriptor_name']
        descriptor_options = {key : val for key, val in ac_config['descriptor'].items() if key != "descriptor_name"}
        acd = getattr(slh.layers.descriptor, descriptor_name)(radial_basis=radial_basis,
                                                              **descriptor_options)
        return acd
    
    def build_bond_centered_descriptor(self):
        bc_config = self.config['bond_centered']
        if bc_config["tensor_module"] == "tensor":
            tensor_module = partial(e3x.nn.Tensor,
                                    param_dtype=getattr(jnp,bc_config["tensor_module_dtype"]))
        elif bc_config["tensor_module"] == "fused_tensor":
            tensor_module = partial(e3x.nn.FusedTensor,
                                    param_dtype=getattr(jnp,bc_config["tensor_module_dtype"]))
        bcd = BondCenteredTensorMomentDescriptor(cutoff=bc_config["cutoff"],
                                                 max_basis_degree=bc_config["max_basis_degree"],
                                                 max_degree=bc_config["max_degree"],
                                                 tensor_module=tensor_module,
                                                 bond_expansion_options=bc_config["bond_expansion_options"])
        return bcd
    
    def build_mlp(self):
        mlp_config = self.config['mlp']
        dense_layer = partial(e3x.nn.Dense, 
                              param_dtype=getattr(jnp,mlp_config["mlp_dtype"]))
        mlp = DenseBlock(dense_layer=dense_layer,
                         layer_widths=mlp_config['mlp_layer_widths'],
                         activation=getattr(e3x.nn, mlp_config["mlp_activation_function"])
                        )
        return mlp

    def build_readout(self, readout_nfeatures: int, max_ell: int):
        return Readout(readout_nfeatures, max_ell=max_ell)
    
    def build_off_diagonal_scale_shift(self, 
                                       exp_prefactors,
                                       exp_lengthscales,
                                       exp_powers):
        if np.any(np.array([exp_prefactors,exp_lengthscales,exp_powers]) == None):  # noqa: E711
            return lambda x: x
        else:
            return OffDiagonalScaleShift(exp_prefactors = exp_prefactors,
                                         exp_lengthscales = exp_lengthscales,
                                         exp_powers = exp_powers)
        
    def build_on_diagonal_scale_shift(self,
                                      shifts,
                                      scales):
        if np.any(np.array(shifts, scales) == None):  # noqa: E711
            return lambda x: x
        else:
            return OnDiagonalScaleShift(shifts = shifts,
                                        scales = scales)
    
    def build_lcao_hamiltonian_model(self, 
                                     readout_nfeatures: int, 
                                     max_ell: int,
                                     exp_prefactors,
                                     exp_lengthscales,
                                     exp_powers,
                                     shifts,
                                     scales
                                    ):
        acd = self.build_atom_centered_descriptor()
        bcd = self.build_bond_centered_descriptor()
        mlp = self.build_mlp()
        off_dro = self.build_readout(readout_nfeatures,
                                     max_ell)
        on_dro = self.build_readout(readout_nfeatures,
                                    max_ell)
        off_dss = self.build_off_diagonal_scale_shift(exp_prefactors,
                                                      exp_lengthscales,
                                                      exp_powers)
        on_dss = self.build_on_diagonal_scale_shift(shifts, 
                                                    scales)
        
        hmodel = HamiltonianModel(atom_centered=acd,
                                  bond_centered=bcd,
                                  dense=mlp,
                                  off_diag_readout=off_dro,
                                  on_diag_readout=on_dro,
                                  off_diag_scale_shift = off_dss,
                                  on_diag_scale_shift = on_dss
                                 )
        
        return hmodel
