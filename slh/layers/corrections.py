import flax.linen as nn
import jax
import jax.numpy as jnp


class ExponentialScaleCorrection(nn.Module):
  nfeatures: int
  max_ell: int

  def setup(self):
    self.prefactors = self.param("prefactors",
                                 nn.initializers.constant(27.0),
                                 shape=(1, 2, self.max_ell + 1, self.nfeatures))
    self.exponents = self.param("exponents",
                                nn.initializers.constant(0.7),
                                shape=(1, 2, self.max_ell + 1, self.nfeatures))
    self.offsets = self.param("offsets",
                                nn.initializers.constant(0.1),
                                shape=(1, 2, self.max_ell + 1, self.nfeatures))

  def __call__(self, neighbour_distances, irreps):
    with jax.ensure_compile_time_eval():
      repeats = 2 * jnp.arange(self.max_ell + 1) + 1

    correctly_shaped_prefactors = jnp.repeat(nn.softplus(self.prefactors), repeats=repeats, axis=-2, total_repeat_length=(self.max_ell + 1) ** 2)
    correctly_shaped_exponents = jnp.repeat(nn.softplus(self.exponents), repeats=repeats, axis=-2, total_repeat_length=(self.max_ell + 1) ** 2)
    correctly_shaped_offsets = jnp.repeat(nn.softplus(self.offsets), repeats=repeats, axis=-2, total_repeat_length=(self.max_ell + 1) ** 2)

    assert correctly_shaped_exponents.shape[-1] == self.nfeatures
    assert correctly_shaped_exponents.shape[-2] == (self.max_ell + 1)**2

    assert correctly_shaped_prefactors.shape[-1] == self.nfeatures
    assert correctly_shaped_prefactors.shape[-2] == (self.max_ell + 1)**2

    assert correctly_shaped_offsets.shape[-1] == self.nfeatures
    assert correctly_shaped_offsets.shape[-2] == (self.max_ell + 1)**2

    return irreps * (correctly_shaped_prefactors * jnp.exp( - jnp.einsum('l..., pl -> p...', correctly_shaped_exponents, neighbour_distances)) + correctly_shaped_offsets)
