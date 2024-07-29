import jax.numpy as jnp
import optax


def huber_loss(h_irreps_off_diagonal_predicted,
               h_irreps_on_diagonal_predicted, 
               batch_labels,
               loss_weights = {"on_diagonal" : 0.4,
                               "off_diagonal" : 1.6
                              }
              ):

    loss_weight_sum = sum(loss_weights.values())
    on_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_on_diagonal_predicted, batch_labels["h_irreps_on_diagonal"]),
                    where=batch_labels["mask_on_diagonal"])

    on_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_on_diagonal_predicted - batch_labels["h_irreps_on_diagonal"]),
                        where=batch_labels["mask_on_diagonal"])

    off_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_off_diagonal_predicted, batch_labels["h_irreps_off_diagonal"]),
                    where=batch_labels["mask_off_diagonal"])

    off_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_off_diagonal_predicted - batch_labels["h_irreps_off_diagonal"]),
                        where=batch_labels["mask_off_diagonal"])
    
    weighted_mean_loss = (loss_weights['on_diagonal']*on_diagonal_loss + \
                          loss_weights['off_diagonal']*off_diagonal_loss ) / loss_weight_sum
    weighted_mean_mae_loss = (loss_weights['on_diagonal']*on_diagonal_mae_loss + \
                              loss_weights['off_diagonal']*off_diagonal_mae_loss ) / loss_weight_sum
    return 5 * weighted_mean_loss + 5 * weighted_mean_mae_loss, weighted_mean_mae_loss, off_diagonal_mae_loss, on_diagonal_mae_loss
