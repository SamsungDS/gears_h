from typing import Union

import e3x
import jax.numpy as jnp
from jaxtyping import Array, Float
import optax

def weighted_mse_and_rmse(h_irreps_off_diagonal_predicted,
                          h_irreps_on_diagonal_predicted, 
                          batch_labels,
                          loss_parameters
                         ):
    off_diagonal_weight = loss_parameters['off_diagonal_weight']
    on_diagonal_weight = loss_parameters['on_diagonal_weight']
    mse_weight = loss_parameters['mse_weight']
    rmse_weight = loss_parameters['rmse_weight']
    loss_multiplier = loss_parameters['loss_multiplier']
    loss_weight_sum = off_diagonal_weight + on_diagonal_weight

    on_diagonal_mse = jnp.mean(optax.squared_error(h_irreps_on_diagonal_predicted, batch_labels["h_irreps_on_diagonal"]),
                                where=batch_labels["mask_on_diagonal"])
    on_diagonal_rmse = jnp.sqrt(on_diagonal_mse)
    off_diagonal_mse = jnp.mean(optax.squared_error(h_irreps_off_diagonal_predicted, batch_labels["h_irreps_off_diagonal"]),
                                 where=batch_labels["mask_off_diagonal"])
    off_diagonal_rmse = jnp.sqrt(off_diagonal_mse)
    
    weighted_mse = (on_diagonal_weight*on_diagonal_mse + \
                    off_diagonal_weight*off_diagonal_mse ) / loss_weight_sum
    weighted_rmse = (on_diagonal_weight*on_diagonal_rmse + \
                     off_diagonal_weight*off_diagonal_rmse ) / loss_weight_sum
    weighted_loss = (mse_weight*weighted_mse + rmse_weight*weighted_rmse)/(mse_weight+rmse_weight)

    on_diagonal_mae = jnp.mean(jnp.abs(h_irreps_on_diagonal_predicted - batch_labels["h_irreps_on_diagonal"]),
                               where=batch_labels["mask_on_diagonal"])
    off_diagonal_mae = jnp.mean(jnp.abs(h_irreps_off_diagonal_predicted - batch_labels["h_irreps_off_diagonal"]),
                                where=batch_labels["mask_off_diagonal"])
    weighted_mean_mae = (on_diagonal_weight*on_diagonal_mae + \
                         off_diagonal_weight*off_diagonal_mae ) / loss_weight_sum
    
    return (loss_multiplier * weighted_loss,
            weighted_mean_mae,
            off_diagonal_mae,
            on_diagonal_mae)

def _normalize_error(error_array: Union[Float[Array, '... 1 (max_ell+1)**2 nfeatures'],
                                        Float[Array, '... 2 (max_ell+1)**2 nfeatures']], 
                     reference_array: Union[Float[Array, '... 1 (max_ell+1)**2 nfeatures'],
                                            Float[Array, '... 2 (max_ell+1)**2 nfeatures']]):
    max_ell = e3x.nn.features._extract_max_degree_and_check_shape(reference_array.shape)
    normalized_error = jnp.zeros(error_array.shape)
    for ell in range(max_ell + 1):
        error_array_slice = error_array[..., ell**2 : (ell + 1)**2 ,:]
        reference_array_slice = reference_array[..., ell**2 : (ell + 1)**2 ,:]
        reference_array_slice_norm = e3x.ops.norm(reference_array_slice, axis=-2, keepdims=True)
        # Replace small values in the norm with 1 where they are small enough to make things numerically unstable
        reference_array_slice_norm = jnp.where(reference_array_slice_norm*reference_array_slice_norm > jnp.finfo(error_array.dtype).tiny,
                                               reference_array_slice_norm,
                                               1)
        reference_array_slice_norm = jnp.clip(reference_array_slice_norm, min = 1e-2, max = 1.0)
        normalized_error = normalized_error.at[..., ell**2 : (ell + 1)**2 ,:].set(error_array_slice / reference_array_slice_norm)
    
    return normalized_error

def irrep_scaled_loss(h_irreps_off_diagonal_predicted,
                      h_irreps_on_diagonal_predicted, 
                      batch_labels,
                      loss_parameters
                     ):
    off_diagonal_weight = loss_parameters['off_diagonal_weight']
    on_diagonal_weight = loss_parameters['on_diagonal_weight']
    mse_weight = loss_parameters['mse_weight']
    rmse_weight = loss_parameters['rmse_weight']
    loss_multiplier = loss_parameters['loss_multiplier']
    loss_weight_sum = off_diagonal_weight + on_diagonal_weight

    off_diagonal_error = h_irreps_off_diagonal_predicted - batch_labels['h_irreps_off_diagonal']
    normalized_off_diagonal_error = _normalize_error(off_diagonal_error, batch_labels['h_irreps_off_diagonal'])
    on_diagonal_error = h_irreps_on_diagonal_predicted - batch_labels['h_irreps_on_diagonal']
    normalized_on_diagonal_error = _normalize_error(on_diagonal_error, batch_labels['h_irreps_on_diagonal'])

    off_diagonal_mse = jnp.mean(optax.squared_error(normalized_off_diagonal_error),
                                where=batch_labels["mask_off_diagonal"])
    on_diagonal_mse = jnp.mean(optax.squared_error(normalized_on_diagonal_error),
                                where=batch_labels["mask_on_diagonal"])
    
    off_diagonal_rmse = jnp.sqrt(off_diagonal_mse)
    on_diagonal_rmse = jnp.sqrt(on_diagonal_mse)

    weighted_mse = (on_diagonal_weight*on_diagonal_mse + \
                    off_diagonal_weight*off_diagonal_mse ) / loss_weight_sum
    weighted_rmse = (on_diagonal_weight*on_diagonal_rmse + \
                     off_diagonal_weight*off_diagonal_rmse ) / loss_weight_sum
    weighted_loss = (mse_weight*weighted_mse + rmse_weight*weighted_rmse)/(mse_weight+rmse_weight)

    on_diagonal_mae = jnp.mean(jnp.abs(h_irreps_on_diagonal_predicted - batch_labels["h_irreps_on_diagonal"]),
                               where=batch_labels["mask_on_diagonal"])
    off_diagonal_mae = jnp.mean(jnp.abs(h_irreps_off_diagonal_predicted - batch_labels["h_irreps_off_diagonal"]),
                                where=batch_labels["mask_off_diagonal"])
    weighted_mean_mae = (on_diagonal_weight*on_diagonal_mae + \
                         off_diagonal_weight*off_diagonal_mae ) / loss_weight_sum
    
    return (loss_multiplier * weighted_loss,
            weighted_mean_mae,
            off_diagonal_mae,
            on_diagonal_mae)