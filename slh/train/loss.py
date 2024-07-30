import jax.numpy as jnp
import optax


def weighted_huber_and_mae(h_irreps_off_diagonal_predicted,
                           h_irreps_on_diagonal_predicted, 
                           batch_labels,
                           loss_parameters
                          ):
    off_diagonal_weight = loss_parameters['off_diagonal_weight']
    on_diagonal_weight = loss_parameters['on_diagonal_weight']
    huber_weight = loss_parameters['huber_weight']
    mae_weight = loss_parameters['mae_weight']
    loss_multiplier = loss_parameters['loss_multiplier']
    loss_weight_sum = off_diagonal_weight + on_diagonal_weight

    on_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_on_diagonal_predicted, batch_labels["h_irreps_on_diagonal"]),
                                where=batch_labels["mask_on_diagonal"])

    on_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_on_diagonal_predicted - batch_labels["h_irreps_on_diagonal"]),
                                    where=batch_labels["mask_on_diagonal"])

    off_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_off_diagonal_predicted, batch_labels["h_irreps_off_diagonal"]),
                                 where=batch_labels["mask_off_diagonal"])

    off_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_off_diagonal_predicted - batch_labels["h_irreps_off_diagonal"]),
                                     where=batch_labels["mask_off_diagonal"])
    
    weighted_mean_loss = (on_diagonal_weight*on_diagonal_loss + \
                          off_diagonal_weight*off_diagonal_loss ) / loss_weight_sum
    weighted_mean_mae_loss = (on_diagonal_weight*on_diagonal_mae_loss + \
                              off_diagonal_weight*off_diagonal_mae_loss ) / loss_weight_sum
    
    weighted_combined_loss = huber_weight*weighted_mean_loss + mae_weight*weighted_mean_mae_loss / (huber_weight + mae_weight)
    
    return (loss_multiplier * weighted_combined_loss, 
            weighted_mean_mae_loss, 
            off_diagonal_mae_loss, 
            on_diagonal_mae_loss)
