from functools import partial

import e3x
import jax
import jax.numpy as jnp

import slh
from slh.config.train_config import ModelConfig
from slh.layers.descriptor import (
    BondCenteredTensorMomentDescriptor,
    SAAtomCenteredDescriptor,
    SpeciesAwareRadialBasis,
    TDSAAtomCenteredDescriptor
)
from slh.layers.readout import Readout
from slh.layers.residual_dense import DenseBlock
from slh.model.hmodel import HamiltonianModel


def build_model(config, readout_config):
    radial_descriptor = SpeciesAwareRadialBasis(
        cutoff=7.0, num_radial=32, max_degree=2, name="radial basis"
    )

    atom_centered_descriptor = SAAtomCenteredDescriptor(
        radial_basis=radial_descriptor,
    )

    bond_descriptor = BondCenteredTensorMomentDescriptor(
        cutoff=7.0, name="bond descriptor"
    )

    neural_net = DenseBlock()

    readout = Readout(
        features=5,
    )

class ModelBuilder:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def build_species_aware_radial_basis(self):
        radial_config = self.config.atom_centered.radial_basis
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
        ac_config = self.config.atom_centered
        descriptor_name = ac_config.descriptor.descriptor_name
        descriptor_options = {key : val for key, val in ac_config.descriptor.items() if key != descriptor_name}
        acd = getattr(slh.layers.descriptor, descriptor_name)(**descriptor_options)
        return acd
    
    def build_bond_centered_descriptor(self):
        bc_config = self.config.bond_centered
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

    

# class ModelBuilder:
#     def __init__(self, model_config: ModelConfig, n_species: int = 119):
#         self.config = model_config
#         self.n_species = n_species

#     def build_basis_function(self):
#         basis_fn = GaussianBasis(
#             n_basis=self.config["n_basis"],
#             r_min=self.config["r_min"],
#             r_max=self.config["r_max"],
#             dtype=self.config["descriptor_dtype"],
#         )
#         return basis_fn

#     def build_radial_function(self):
#         basis_fn = self.build_basis_function()
#         radial_fn = RadialFunction(
#             n_radial=self.config["n_radial"],
#             basis_fn=basis_fn,
#             n_species=self.n_species,
#             emb_init=self.config["emb_init"],
#             dtype=self.config["descriptor_dtype"],
#         )
#         return radial_fn

#     def build_descriptor(
#         self,
#         apply_mask,
#     ):
#         radial_fn = self.build_radial_function()
#         descriptor = GaussianMomentDescriptor(
#             radial_fn=radial_fn,
#             n_contr=self.config["n_contr"],
#             dtype=self.config["descriptor_dtype"],
#             apply_mask=apply_mask,
#         )
#         return descriptor

#     def build_readout(self):
#         readout = AtomisticReadout(
#             units=self.config["nn"],
#             b_init=self.config["b_init"],
#             dtype=self.config["readout_dtype"],
#         )
#         return readout

#     def build_scale_shift(self, scale, shift):
#         scale_shift = PerElementScaleShift(
#             n_species=self.n_species,
#             scale=scale,
#             shift=shift,
#             dtype=self.config["scale_shift_dtype"],
#         )
#         return scale_shift

#     def build_atomistic_model(
#         self,
#         scale,
#         shift,
#         apply_mask,
#     ):
#         descriptor = self.build_descriptor(apply_mask)
#         readout = self.build_readout()
#         scale_shift = self.build_scale_shift(scale, shift)

#         atomistic_model = AtomisticModel(descriptor, readout, scale_shift)
#         return atomistic_model

#     def build_energy_model(
#         self,
#         scale=1.0,
#         shift=0.0,
#         apply_mask=True,
#         init_box: np.array = np.array([0.0, 0.0, 0.0]),
#         inference_disp_fn=None,
#     ):
#         atomistic_model = self.build_atomistic_model(
#             scale,
#             shift,
#             apply_mask,
#         )
#         corrections = []
#         if self.config["use_zbl"]:
#             repulsion = ZBLRepulsion(
#                 apply_mask=apply_mask,
#                 r_max=self.config["r_max"],
#             )
#             corrections.append(repulsion)

#         model = EnergyModel(
#             atomistic_model,
#             corrections=corrections,
#             init_box=init_box,
#             inference_disp_fn=inference_disp_fn,
#         )
#         return model

#     def build_energy_derivative_model(
#         self,
#         scale=1.0,
#         shift=0.0,
#         apply_mask=True,
#         init_box: np.array = np.array([0.0, 0.0, 0.0]),
#         inference_disp_fn=None,
#     ):
#         energy_model = self.build_energy_model(
#             scale,
#             shift,
#             apply_mask,
#             init_box=init_box,
#             inference_disp_fn=inference_disp_fn,
#         )

#         model = EnergyDerivativeModel(
#             energy_model,
#             calc_stress=self.config["calc_stress"],
#         )
#         return model


def build_lcao_hamiltonian_model(dummy):
    model = HamiltonianModel(
        atom_centered=(
            SAAtomCenteredDescriptor(
                SpeciesAwareRadialBasis(
                    cutoff=6.8,
                    max_degree=2,
                    num_elemental_embedding=8,
                    num_radial=16,
                    tensor_module=partial(e3x.nn.Tensor, param_dtype=jnp.bfloat16),
                    embedding_residual_connection=True,
                ),
                use_fused_tensor=False,
                embedding_residual_connection=True,
            )
        ),
        bond_centered=(
            BondCenteredTensorMomentDescriptor(
                cutoff=6.8,
                max_actp_degree=4,
                max_basis_degree=4,
                max_degree=4,
                tensor_module=partial(e3x.nn.Tensor, param_dtype=jnp.bfloat16),
            )
        ),
        dense=DenseBlock(
            dense_layer=partial(e3x.nn.Dense, param_dtype=jnp.bfloat16),
            layer_widths=[128],
        ),
        readout=Readout(2, max_ell=2),
    )
    return model
