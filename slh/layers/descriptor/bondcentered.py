from dataclasses import field
from functools import partial
from typing import Union

import e3x
import flax.linen as nn
import jax.numpy as jnp

from slh.layers.layer_norm import LayerNorm


class BondCenteredTensorMomentDescriptor(nn.Module):
    cutoff: float
    max_actp_degree: int = 4
    max_basis_degree: int = 2
    max_degree: int = 4
    tensor_module: Union[e3x.nn.Tensor, e3x.nn.FusedTensor] = e3x.nn.FusedTensor
    bond_expansion_options: dict = field(default_factory=lambda: {})
    # radial_function = e3x.nn.basic_fourier

    def setup(self):
        radial_kwargs = self.bond_expansion_options.pop("radial_kwargs")
        cutoff_kwargs = self.bond_expansion_options.pop("cutoff_kwargs")
        radial_function = self.bond_expansion_options.pop("radial_fn")
        cutoff_function = self.bond_expansion_options.pop("cutoff_fn")
        self.bond_expansion_options['max_degree'] = self.max_basis_degree
        self.bond_expansion = partial(e3x.nn.basis,
                                      radial_fn = partial(getattr(e3x.nn, radial_function),
                                                          **radial_kwargs),
                                      cutoff_fn = partial(getattr(e3x.nn, cutoff_function),
                                                          **cutoff_kwargs),
                                      cartesian_order = False,
                                      **self.bond_expansion_options
                                     )

    @nn.compact
    def __call__(self, atomic_descriptors, neighbour_indices, neighbour_displacements):
        
        neighbours_i, neighbours_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        num_radial_features = atomic_descriptors.shape[-1]

        atom1_desc, atom2_desc = (
            atomic_descriptors[neighbours_i],
            atomic_descriptors[neighbours_j],
        )

        # TODO tunable number of layers/widths/etc.
        # TODO also residual on the first dense so we're consistent with
        # atomcentered
        y = e3x.nn.add(atom1_desc, atom2_desc)
        y = e3x.nn.Dense(num_radial_features)(y)
        y = LayerNorm()(y)
        y = e3x.nn.mish(y)
        y = e3x.nn.Dense(num_radial_features)(y) + y

        # We put in information about the orientation/length of the bond vector here
        bond_expansion = self.bond_expansion(neighbour_displacements).astype(jnp.float32)
        bond_expanded_dense = e3x.nn.Dense(num_radial_features)(bond_expansion)

        # num_pairs x 2 x (max_degree + 1)^2 x num_radial_features
        tp = self.tensor_module(
            max_degree=self.max_degree, cartesian_order=False, dtype=jnp.float32
        )
        y = tp(bond_expanded_dense, y)

        return y
