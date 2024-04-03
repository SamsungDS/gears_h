import e3x
import jax
import flax.linen as nn

from jaxtyping import Float, Array, Int

from surrogatelcaohamiltonians.layers.descriptor.radial_basis import (
    SpeciesAwareRadialBasis,
)

e3x.Config.set_cartesian_order(False)


class AtomCenteredTensorMomentDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    num_moment_features: int = 64  # TODO this can in principle be a list of ints
    max_moment: int = 2
    moment_max_degree: int = 4
    tensor_module: nn.Module = e3x.nn.TensorDense
    embedding_residual_connection: bool = True

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.radial_basis.num_radial, name="transform embedding"
        )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Float[Array, "..."],
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
        neighbour_indices: Int[Array, "... 2 num_neighbours"],
    ):
        """_summary_

        Parameters
        ----------
        atomic_numbers : _type_
            _description_
        neighbour_displacements : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        idx_i, idx_j = neighbour_indices[0], neighbour_indices[1]
        Z_i, Z_j = atomic_numbers[idx_i], atomic_numbers[idx_j]

        y = self.radial_basis(neighbour_displacements=neighbour_displacements, Z_j=Z_j)
        # The line above is our 'first moment' since we have directional information.
        # Each subsequent moment happens down here.
        # The moments down here are 'even' in that it's y x y = yy,
        # then yy x yy = yyyy, and so on. We keep it this way until I can figure out if
        # y x yy = yy x y upto a difference in weights.
        # TODO: This WILL error out if your first y doesn't have a high enough degree
        # to tensor onto moment_max_degree. 
        for _ in range(self.max_moment - 1):
            y = self.tensor_module(
                features=self.num_moment_features, max_degree=self.moment_max_degree
            )(y)

        # y is currently n_neighbours x 2 x (basis_max_degree + 1)**2 x num_basis_features

        transformed_embedding = self.embedding_transformation(self.embedding(Z_i))

        # This is currently num_pairs x 2 x (moment_max_degree + 1)^2 x basis
        y = e3x.nn.Tensor(max_degree=self.moment_max_degree, name="emb x basis")(
            transformed_embedding, y
        )

        # Contract over all neighbours of atoms indexed by idx_i
        # This is the ONLY "message-passing" step.
        # This is now num_atoms x 2 x (moment_max_degree + 1)^2 x basis
        y = e3x.ops.indexed_sum(y, dst_idx=idx_i, num_segments=len(atomic_numbers))

        # Do less math by doing the residual connectins here.
        if self.embedding_residual_connection:
            y = e3x.nn.add(
                y, self.embedding_transformation(self.embedding(atomic_numbers))
            )

        # TODO In principle we might want a residual connection here of y for training
        # stability reasons, but we can also do that later.
        return y
