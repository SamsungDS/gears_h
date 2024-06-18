from dataclasses import field
from functools import partial

import e3x
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from slh.layers.descriptor.radial_basis import (SpeciesAwareRadialBasis,
                                                jinclike)


class AtomCenteredTensorMomentDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    num_moment_features: int = 64  # TODO this can in principle be a list of ints
    max_moment: int = 2
    moment_max_degree: int = 4
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.num_moment_features * self.max_moment + self.radial_basis.num_radial,
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

        num_neighbour_normalization = self.param(
            "neighbour_normalization",
            nn.initializers.constant(len(neighbour_indices) / len(atomic_numbers)),
            1,
            dtype=jnp.float32,
        )

        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        # Do the 2-body summation of things first
        y = (
            e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))
            / num_neighbour_normalization
        )

        # This will error out if your first y doesn't have a high enough degree
        # to tensor onto moment_max_degree.
        assert y.dtype == jnp.float32
        ylist = [y]
        for i in range(self.max_moment):
            tmp = e3x.nn.TensorDense(
                features=self.num_moment_features,
                max_degree=self.moment_max_degree,
                use_fused_tensor=self.use_fused_tensor,
                cartesian_order=False,
                dtype=jnp.float32,
                name=f"ac_td_{i}",
            )(ylist[-1])

            ylist.append(tmp.astype(jnp.float32))

        ylist = [
            e3x.nn.features.change_max_degree_or_type(
                y, max_degree=self.moment_max_degree, include_pseudotensors=True
            )
            for y in ylist
        ]

        y = jnp.concat(ylist, axis=-1)

        transformed_embedding = self.embedding_transformation(
            self.embedding(atomic_numbers)
        )

        # This is currently num_padded_atoms x 2 x (moment_max_degree + 1)^2 x basis
        y = e3x.nn.FusedTensor(
            max_degree=self.moment_max_degree,
            name="ac emb x basis",
            cartesian_order=False,
        )(transformed_embedding, y)

        assert y.dtype == jnp.float32

        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y + e3x.nn.mish(y)


class MPAtomCenteredDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.radial_basis.num_radial,
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

        num_neighbour_normalization = self.param(
            "neighbour_normalization",
            nn.initializers.constant(
                (len(neighbour_indices) / len(atomic_numbers)) ** 0.5
            ),
            1,
            dtype=jnp.float32,
        )

        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        for _ in range(2):
            y = (
                e3x.nn.MessagePass(
                    max_degree=3,
                    use_basis_bias=True,
                    cartesian_order=False,
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
                / num_neighbour_normalization
            )

            y = e3x.nn.Dense(features=y.shape[-1])(y) + y

        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y  # + e3x.nn.mish(y)


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
