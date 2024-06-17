import flax.linen as nn
import jax
import jax.numpy as jnp


class ExponentialScaleCorrection(nn.Module):
    nfeatures: int
    max_ell: int

    def setup(self):
        self.prefactors = self.param(
            "prefactors",
            nn.initializers.constant(10.0),
            shape=(1, 2, self.max_ell + 1, self.nfeatures),
        )
        self.exponents = self.param(
            "exponents",
            nn.initializers.constant(1.4),
            shape=(1, 2, self.max_ell + 1, self.nfeatures),
        )

    def __call__(self, neighbour_distances):
        with jax.ensure_compile_time_eval():
            repeats = 2 * jnp.arange(self.max_ell + 1) + 1

        correctly_shaped_prefactors = jnp.repeat(
            1.0 + nn.softplus(self.prefactors),
            repeats=repeats,
            axis=-2,
            total_repeat_length=(self.max_ell + 1) ** 2,
        )
        correctly_shaped_exponents = jnp.repeat(
            jnp.log(nn.softplus(self.exponents)),
            repeats=repeats,
            axis=-2,
            total_repeat_length=(self.max_ell + 1) ** 2,
        )

        assert correctly_shaped_exponents.shape[-1] == self.nfeatures
        assert correctly_shaped_exponents.shape[-2] == (self.max_ell + 1) ** 2

        assert correctly_shaped_prefactors.shape[-1] == self.nfeatures
        assert correctly_shaped_prefactors.shape[-2] == (self.max_ell + 1) ** 2

        return (
            correctly_shaped_prefactors
            * jnp.exp(
                -jnp.einsum(
                    "l..., pl -> p...", correctly_shaped_exponents, neighbour_distances
                )
            )
        )
