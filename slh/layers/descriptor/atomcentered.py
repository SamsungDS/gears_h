from functools import partial

import e3x
import jax.numpy as jnp
import flax.linen as nn

from jaxtyping import Float, Array, Int

from slh.layers.descriptor.radial_basis import (
    SpeciesAwareRadialBasis,
)


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

            # tmp = e3x.nn.FusedTensor(
            #     max_degree=self.moment_max_degree,
            #     cartesian_order=False,
            #     name=f"ac_ft_{i}",
            # )(y, ylist[-1])

            ylist.append(tmp.astype(jnp.float32))

        ylist = [
            e3x.nn.features.change_max_degree_or_type(
                y, max_degree=self.moment_max_degree, include_pseudotensors=True
            )
            for y in ylist
        ]

        y = jnp.concat(ylist, axis=-1)

        # transformed_embedding = self.embedding_transformation(self.embedding(Z_i))
        transformed_embedding = self.embedding_transformation(
            self.embedding(atomic_numbers)
        )

        # This is currently num_pairs x 2 x (moment_max_degree + 1)^2 x basis
        y = e3x.nn.FusedTensor(
            max_degree=self.moment_max_degree,
            name="ac emb x basis",
            cartesian_order=False,
        )(transformed_embedding, y)

        # y = e3x.nn.add(y.astype(jnp.float32), transformed_embedding.astype(jnp.float32))
        assert y.dtype == jnp.float32

        # Do less math by doing the residual connectins here.
        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y + e3x.nn.mish(y)
