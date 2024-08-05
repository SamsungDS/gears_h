import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

class LayerNorm(nn.module):
    shape: e3x.nn.features.Shape

    def setup(self):
        self.max_ell = e3x.nn.features._extract_max_degree_and_check_shape(self.shape)
        self.nfeatures = self.shape[-1]
        self.idx_l = e3x.nn.modules._duplication_indices_for_max_degree(self.max_ell)
        # 1 x P x L x F
        self.irrepwise_scaling = self.param("irrepwise_scaling",
                                            nn.initializers.constant(1.0),
                                            shape = (1, 1, self.max_ell+1, self.nfeatures)
                                           )
        self.irrepwise_scaling = jnp.take(self.irrepwise_scaling,self.idx_l,axis=-2)
        # TODO how and where to do the normalization--do we do it here in the setup once, or every time we call?
        
    def __call__(self, feature_array: Float[Array, '... 1 (max_ell+1)**2 nfeatures']):
        # normalization
        normalized_feature_array = jnp.zeros(self.shape)
        for i in range(self.max_ell+1):
            indices = jnp.where(self.idx_l == i)
            feature_array_slice = feature_array[...,indices,:]
            normalized_feature_array_slice = e3x.ops.normalize(feature_array_slice, axis=-2)
            normalized_feature_array = normalized_feature_array.at[...,indices,:].set(normalized_feature_array_slice)
        
        return self.irrepwise_scaling*normalized_feature_array