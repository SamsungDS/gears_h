from functools import partial

import e3x
import jax.numpy as jnp

import slh
from slh.config.train_config import ModelConfig
from slh.layers.descriptor import (
    BondCenteredTensorMomentDescriptor,
    SpeciesAwareRadialBasis
)
from slh.layers.readout import Readout
from slh.layers.residual_dense import DenseBlock
from slh.model.hmodel import HamiltonianModel

class ModelBuilder:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def build_species_aware_radial_basis(self):
        radial_config = self.config['atom_centered']['radial_basis']
        if radial_config["tensor_module"] == "tensor":
            tensor_module = partial(e3x.nn.Tensor,
                                    param_dtype=getattr(jnp,radial_config["tensor_module_dtype"]))
        elif radial_config["tensor_module"] == "fused_tensor":
            tensor_module = partial(e3x.nn.FusedTensor,
                                    param_dtype=getattr(jnp,radial_config["tensor_module_dtype"]))
        sarb = SpeciesAwareRadialBasis(cutoff=radial_config["cutoff"],
                                       num_radial=radial_config["num_radial"],
                                       max_degree=radial_config["max_degree"],
                                       num_elemental_embedding=radial_config["num_elemental_embedding"],
                                       tensor_module=tensor_module,
                                       embedding_residual_connection=radial_config["embedding_residual_connection"] 
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
                                                 max_actp_degree=bc_config["max_actp_degree"],
                                                 max_basis_degree=bc_config["max_basis_degree"],
                                                 max_degree=bc_config["max_degree"],
                                                 tensor_module=tensor_module)
        return bcd
    
    def build_mlp(self):
        mlp_config = self.config['mlp']
        dense_layer = partial(e3x.nn.Dense, 
                              param_dtype=getattr(jnp,mlp_config["mlp_dtype"]))
        mlp = DenseBlock(dense_layer=dense_layer,
                         layer_widths=mlp_config['mlp_layer_widths']
                        )
        return mlp

    def build_readout(self, readout_nfeatures: int, max_ell: int):
        return Readout(readout_nfeatures, max_ell=max_ell)
    
    def build_lcao_hamiltonian_model(self, readout_nfeatures: int, max_ell: int):
        acd = self.build_atom_centered_descriptor()
        bcd = self.build_bond_centered_descriptor()
        mlp = self.build_mlp()
        readout = self.build_readout(readout_nfeatures, max_ell=max_ell)
        hmodel = HamiltonianModel(atom_centered=acd,
                                  bond_centered=bcd,
                                  dense=mlp,
                                  readout=readout
                                 )
        return hmodel