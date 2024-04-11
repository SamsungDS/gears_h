from slh.layers.descriptor.radial_basis import (
    SpeciesAwareRadialBasis,
)

import jax
import jax.numpy as jnp


def test_rb():
    disps = jax.random.uniform(jax.random.PRNGKey(42), shape=(4, 3))
    Z_j = jnp.array([4, 5, 5, 5])
    module = SpeciesAwareRadialBasis(2.0, num_radial=32)
    params = module.init(jax.random.PRNGKey(56), disps, Z_j)
    out1 = module.apply(params, disps, jnp.array([5, 5, 4, 4]))
    out2 = module.apply(params, disps, jnp.array([5, 5, 4, 5]))
    for i in range(3):
        assert jnp.allclose(out1[i], out2[i])
    assert not jnp.allclose(out1[3], out2[3])
    # assert not jnp.allclose(out1, out2)
