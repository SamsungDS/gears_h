from dataclasses import field
from functools import partial

import e3x
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gears_h.layers.descriptor.radial_basis import SpeciesAwareRadialBasis
from gears_h.layers.layer_norm import LayerNorm

class ShallowTDSAAtomCenteredDescriptor(nn.Module):
    radial_basis: SpeciesAwareRadialBasis
    num_tensordenses: int
    max_tensordense_degree: int
    num_tensordense_features: int
    use_fused_tensor: bool = False

    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = field(default_factory=lambda: {})
    mp_basis_options: dict = field(default_factory=lambda: {})

    def setup(self):
        self.embedding = self.radial_basis.embedding
        self.embedding_transformation = e3x.nn.Dense(
            self.radial_basis.num_radial + self.num_tensordenses * self.num_tensordense_features,
            name="embed_transform",
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.mp_block: e3x.nn.SelfAttention = partial(e3x.nn.SelfAttention,
            max_degree=self.mp_degree,
            cartesian_order=False,
            use_basis_bias=False,
            use_fused_tensor=self.use_fused_tensor,
            **self.mp_options,
        )

        options = self.mp_basis_options

        options, radial_kwargs = options.pop("radial_kwargs")
        options, radial_function = options.pop("radial_fn")
        options, cutoff_function = options.pop("cutoff_fn")

        self.mp_basis = partial(e3x.nn.basis,
                                radial_fn = partial(getattr(e3x.nn, radial_function),
                                                    **radial_kwargs),
                                cutoff_fn = partial(getattr(e3x.nn, cutoff_function),
                                                    cutoff=self.radial_basis.cutoff),
                                cartesian_order = False,
                                **options
                                )

    @nn.compact
    def __call__(
        self,
        atomic_numbers: Int[Array, " num_atoms"],
        neighbour_indices: Int[Array, "... num_neighbours 2"],        
        neighbour_displacements: Float[Array, "... num_neighbours 3"],
    ):

        idx_i, idx_j = neighbour_indices[:, 0], neighbour_indices[:, 1]
        Z_j = atomic_numbers[idx_j]

        # This is aware of the Z_j's
        y_2b = self.radial_basis(
            neighbour_displacements=neighbour_displacements, Z_j=Z_j
        ).astype(jnp.float32)
        
        y_2b = e3x.ops.indexed_sum(y_2b, dst_idx=idx_i, num_segments=len(atomic_numbers))
        y = [y_2b]

        # Set max degree for the TD to the max achievable degree from the inputs.
        max_td_deg = min(self.max_tensordense_degree, 
                         2 * self.radial_basis.max_degree)
        # Apply TensorDenses
        for i in range(self.num_tensordenses):
            td = e3x.nn.TensorDense(self.num_tensordense_features,
                                    max_td_deg,
                                    cartesian_order=False,
                                    use_fused_tensor=self.use_fused_tensor,
                                   )(y[i])
            # Compute max degree for the next TD to the max achievable degree from the inputs.
            deg = e3x.nn.features._extract_max_degree_and_check_shape(td.shape)
            max_td_deg = min(self.max_tensordense_degree,
                             deg)
            y.append(td)

        # SelfAttention block
        for yy in y:
            for _ in range(self.mp_steps):
                yy = self.mp_block()(
                    inputs=yy,
                    basis=self.mp_basis(neighbour_displacements),
                    src_idx=idx_j,
                    dst_idx=idx_i,
                    num_segments=len(atomic_numbers),
                    cutoff_value=partial(e3x.nn.smooth_cutoff, cutoff=self.radial_basis.cutoff)(jnp.linalg.norm(neighbour_displacements, axis=1)),
                )
                yy = LayerNorm()(yy)

        # Nonlinear block
        yout = []
        for yy in y:
            y0 = e3x.nn.Dense(yy.shape[-1])(yy)
            yy = LayerNorm()(y0)
            yy = e3x.nn.bent_identity(yy)
            yy = e3x.nn.Dense(yy.shape[-1])(yy) + y0
            yout.append(yy)

        # Combine features into one array.
        y = [e3x.nn.features.change_max_degree_or_type(desc, 
                                                       self.max_tensordense_degree, 
                                                       include_pseudotensors=True) for desc in yout]
        y = jnp.concatenate(y, axis=-1)

        return y
