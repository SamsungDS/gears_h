import e3x
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Union

class LayerNorm(nn.Module):
    
    @nn.compact
    def __call__(self, feature_array: Union[Float[Array, '... 1 (max_ell+1)**2 nfeatures'],
                                            Float[Array, '... 2 (max_ell+1)**2 nfeatures']]):
        # irrepwise learnable parameter initialization
        max_ell = e3x.nn.features._extract_max_degree_and_check_shape(feature_array.shape)
        nfeatures = feature_array.shape[-1]
        idx_l = e3x.nn.modules._duplication_indices_for_max_degree(max_ell)
        irrepwise_scaling = self.param("irrepwise_scaling",
                                       nn.initializers.constant(1.0),
                                       shape = (1, feature_array.shape[-3], max_ell+1, nfeatures)
                                      )
        irrepwise_scaling = jnp.take(irrepwise_scaling,idx_l,axis=-2)

        # normalization
        normalized_feature_array = jnp.zeros(feature_array.shape)
        for ell in range(max_ell+1):
            feature_array_slice = feature_array[..., ell**2 : (ell + 1)**2 ,:]
            # indexing along the angular momentum direction for l = 0 destroys relative magnitudes, so we normalize across the feature direction.
            # this is still equivariant for scalars only.
            if ell == 0:
                feature_array_slice = feature_array_slice.at[..., 0, 0, :].subtract(jnp.mean(feature_array_slice[..., 0, 0, :]))
                normalized_feature_array_slice = e3x.ops.normalize(feature_array_slice, axis=-1)
            # for l != 0,  to preserve equivariance, we normalize in the angular momentum
            # direction (per irrep) to scale all irrep components by the same scalar
            else:
                normalized_feature_array_slice = e3x.ops.normalize(feature_array_slice, axis=-2)
            normalized_feature_array = normalized_feature_array.at[..., ell**2 : (ell + 1)**2 ,:].set(normalized_feature_array_slice)
        
        return irrepwise_scaling * normalized_feature_array
