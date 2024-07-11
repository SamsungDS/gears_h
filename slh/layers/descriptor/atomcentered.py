from dataclasses import field
from functools import partial

import e3x
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from slh.layers.descriptor.radial_basis import (SpeciesAwareRadialBasis,
                                                jinclike)


class SAAtomCenteredDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True
    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = field(default_factory=lambda: {})

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.radial_basis.num_radial,
            name="embed_transform",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.mp_block = e3x.nn.SelfAttention(
            max_degree=self.mp_degree,
            cartesian_order=False,
            **self.mp_options,
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, "num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):
        neighbour_displacements = neighbour_displacements

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        Z_i, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        # num_neighbours x 1 x L x F
        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        # num_atoms x 1 x L x F
        y = e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))

        for _ in range(self.mp_steps):
            y = e3x.nn.SelfAttention(
                max_degree=self.mp_degree,
                use_basis_bias=True,
                cartesian_order=False,
                use_fused_tensor=self.use_fused_tensor,
                num_heads=4,
            )(
                y,
                e3x.nn.basis(
                    neighbour_displacements,
                    max_degree=2,
                    num=16,
                    radial_fn=partial(e3x.nn.sinc, limit=self.radial_basis.cutoff),
                    cartesian_order=False,
                ),
                src_idx=idx_j,
                dst_idx=idx_i,
                num_segments=len(atomic_numbers),
            )

        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y


class TDSAAtomCenteredDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    max_tensordense_degree: int
    num_tensordense_features: int
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.num_tensordense_features,
            name="embed_transform",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, "num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):
        neighbour_displacements = neighbour_displacements

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        Z_i, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        y = e3x.nn.TensorDense(
            self.num_tensordense_features,
            self.max_tensordense_degree,
            cartesian_order=False,
            use_fused_tensor=self.use_fused_tensor,
        )(y)

        for _ in range(2):
            y = e3x.nn.SelfAttention(
                max_degree=4,
                use_basis_bias=True,
                cartesian_order=False,
                # num_heads=2,
                use_fused_tensor=self.use_fused_tensor,
            )(
                y,
                e3x.nn.basis(
                    neighbour_displacements,
                    max_degree=2,
                    num=16,
                    radial_fn=partial(e3x.nn.sinc, limit=self.radial_basis.cutoff),
                    cartesian_order=False,
                ),
                src_idx=neighbour_indices[:, 1],
                dst_idx=neighbour_indices[:, 0],
                num_segments=len(atomic_numbers),
            )

            # y = e3x.nn.Dense(features=y.shape[-1])(y) + y

        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y
