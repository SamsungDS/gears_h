import jax.numpy as jnp
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
