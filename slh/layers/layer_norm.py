import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

class LayerNorm(nn.module):
    nfeatures: Int
    max_ell: Int

    def setup(self):
        # 1 x P x L x F
        self.irrepwise_scaling = self.param("irrepwise_scaling",
                                            nn.initializers.constant(1.0),
                                            shape = (1, 1, self.max_ell+1, self.nfeatures)
                                           )
        
    def __call__(self, feature_array: Float[Array, '... 1 max_ell nfeatures']):
        return self.irrepwise_scaling*feature_array