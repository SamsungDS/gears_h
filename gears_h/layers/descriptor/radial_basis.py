from functools import partial
from typing import Any, Union

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class SpeciesAwareRadialBasis(nn.Module):
    cutoff: float
    num_radial: int = 8
    max_degree: int = 3
    num_elemental_embedding: int = 64

    def setup(self):
        self.radial_function = partial(
                        e3x.nn.basic_fourier,
                        limit=self.cutoff,
                    )

        self.embedding = e3x.nn.Embed(
            83,
            self.num_elemental_embedding,
            name="elem_embed",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

    @nn.compact
    def __call__(
        self,
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
        Z_j: Float[Array, "... num_neighbours"],
    ):
        """Generate 2-body species aware radial basis.

        Parameters
        ----------
        neighbour_displacements: Float[Array, "... num_neighbours 3"]
            Input array containing vectors to neighbours around a given point.
        Z_j : Float[Array, "... num_neighbours"]
            Input array of atomic numbers of the points.

        Returns
        -------
        Float[Array, "... num_neighbors 1 (max_degree + 1)**2 num_radial]
            Species-aware radial basis
        """
        assert neighbour_displacements.dtype == jnp.float32

        basis_expansion = e3x.nn.basis(
            neighbour_displacements,
            num=self.num_radial,
            max_degree=self.max_degree,
            radial_fn=self.radial_function,
            cutoff_fn=partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
            cartesian_order=False,
        ).astype(jnp.float32)

        assert basis_expansion.dtype == jnp.float32

        # We transform the embedding dimension to the radial basis dimension
        # so we can product meaningfully
        transformed_embedding = e3x.nn.Dense(
            self.num_radial, dtype=jnp.float32, name="embed_transform"
        )(self.embedding(Z_j))

        y = basis_expansion * transformed_embedding

        return y.astype(jnp.float32)
