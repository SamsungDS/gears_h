from dataclasses import field
from functools import partial

import e3x
import flax.linen as nn
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, Int
from typing import Literal

from slh.layers.descriptor.radial_basis import SpeciesAwareRadialBasis
from slh.layers.layer_norm import LayerNorm
# from slh.utilities.functions import soft_abs


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
            use_basis_bias=False,
            cartesian_order=False,
            use_fused_tensor=self.use_fused_tensor,
            **self.mp_options,
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, "num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        _, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        # num_neighbours x 1 x L x F
        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        # num_atoms x 1 x L x F
        y = e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))

        # gamma = self.param("gamma", nn.initializers.constant(1.0), shape=(1,))
        
        for _ in range(self.mp_steps):
            y = self.mp_block(
                inputs=y,
                basis=partial(
                    e3x.nn.basis,
                    max_degree=2,
                    num=8,
                    radial_fn=partial(
                        e3x.nn.basic_fourier,
                        limit=self.radial_basis.cutoff,
                    ),
                    cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.radial_basis.cutoff),
                    cartesian_order=False,
                )(neighbour_displacements),
                src_idx=idx_j,
                dst_idx=idx_i,
                num_segments=len(atomic_numbers),
                cutoff_value=partial(e3x.nn.smooth_cutoff, cutoff=self.radial_basis.cutoff)(jnp.linalg.norm(neighbour_displacements, axis=1))
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

    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = field(default_factory=lambda: {})

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.num_tensordense_features,
            name="embed_transform",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.mp_block: e3x.nn.SelfAttention = e3x.nn.SelfAttention(
            max_degree=self.mp_degree,
            cartesian_order=False,
            use_basis_bias=False,
            use_fused_tensor=self.use_fused_tensor,
            **self.mp_options,
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, "num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        _, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        # num_atoms x 1 x L x F
        y = e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))

        for _ in range(self.mp_steps):
            y = e3x.nn.TensorDense(
            self.num_tensordense_features,
            self.max_tensordense_degree,
            cartesian_order=False,
            use_fused_tensor=self.use_fused_tensor,
            )(y)
            y = self.mp_block(
                inputs=y,
                basis=partial(
                    e3x.nn.basis,
                    max_degree=2,
                    num=8,
                    radial_fn=partial(
                        e3x.nn.basic_fourier,
                        limit=self.radial_basis.cutoff,
                    ),
                    cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.radial_basis.cutoff),
                    cartesian_order=False,
                )(neighbour_displacements),
                src_idx=idx_j,
                dst_idx=idx_i,
                num_segments=len(atomic_numbers),
                cutoff_value=partial(e3x.nn.smooth_cutoff, cutoff=self.radial_basis.cutoff)(jnp.linalg.norm(neighbour_displacements, axis=1)),
            )
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )
            y: Array = e3x.ops.normalize(y, axis=-2)

        y = e3x.nn.Dense(self.embedding_transformation.features)(y) + y
        y = LayerNorm()(y)
        y = e3x.nn.mish(y)
        y = e3x.nn.Dense(self.embedding_transformation.features)(y) + y

        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        return y


class MBSAAtomCenteredDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    max_tensordense_degrees: list[int]
    num_tensordense_features: list[int]
    num_mp_steps: list[int]

    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True

    mode: Literal["contiguous", "interleaved"] = "contiguous"

    def setup(self):
        assert len(self.max_tensordense_degrees) == len(self.num_tensordense_features)
        assert len(self.num_mp_steps) <= len(self.num_tensordense_features) + 1

        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.radial_basis.num_radial,
            name="embed_transform",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.mp_block: e3x.nn.SelfAttention = e3x.nn.SelfAttention(
            max_degree=self.mp_degree,
            cartesian_order=False,
            use_basis_bias=False,
            use_fused_tensor=self.use_fused_tensor,
            **self.mp_options,
        )

        self.mp_basis = partial(
            e3x.nn.basis,
            num=8,
            radial_fn=partial(
                e3x.nn.exponential_bernstein,
                limit=self.radial_basis.cutoff,
                gamma=1 / 1.5,
                cuspless=True,
            ),
            cartesian_order=False,
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, "num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        Z_i, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        # This is aware of the Z_j's
        y = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)

        # num_atoms x 1 x L x F
        y = e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))

        # Need to incorporate information of the species of the current atom.
        # We are doing this only for the 2-body term, but of course we can and should
        # think about other terms
        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        if self.path == "contiguous":
            ylist = [y]

            for td_num_features, td_max_degree in zip(
                self.num_tensordense_features, self.max_tensordense_degrees, strict=True
            ):
                ylist.append(
                    e3x.nn.TensorDense(
                        num_features=td_num_features,
                        max_degree=td_max_degree,
                        cartesian_order=False,
                        use_fused_tensor=self.use_fused_tensor,
                    )(ylist[-1])
                )

            # ylist now has 2-body, 3(ish)-body, 4(ish)-body, ... radial features
            outlist = []

            for i, num_mp in enumerate(self.num_mp_steps):
                out = ylist[i]
                for _ in range(num_mp):
                    out = self.mp_block(
                        out,
                        self.mp_basis(neighbour_displacements),
                        src_idx=idx_j,
                        dst_idx=idx_i,
                        num_segments=len(atomic_numbers),
                    )
                outlist.append(out)

        if self.path == "interleaved":
            outlist = []

            for i, num_mp in enumerate(self.num_mp_steps):
                # For the first cascade, we directly take the 2B basis
                out = outlist[i - 1] if i > 0 else y

                # If we are not at the first cascade, we TD the MP'd basis from the previous iteration
                if i > 0:
                    out = e3x.nn.TensorDense(
                        # The TD index lags the MP index by 1 since the first
                        # MP is just the 2B basis
                        num_features=self.num_tensordense_features[i - 1],
                        max_degree=self.max_tensordense_degrees[i - 1],
                        cartesian_order=False,
                        use_fused_tensor=self.use_fused_tensor,
                    )(out)

                for _ in range(num_mp):
                    out = self.mp_block(
                        out,
                        self.mp_basis(neighbour_displacements),
                        src_idx=idx_j,
                        dst_idx=idx_i,
                        num_segments=len(atomic_numbers),
                    )

                outlist.append(out)

        return outlist
