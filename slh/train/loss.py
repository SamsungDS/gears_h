import jax.numpy as jnp
import optax


def huber_loss(h_irreps_off_diagonal_predicted,
               h_irreps_on_diagonal_predicted, 
               batch_labels,
               loss_weights = {"on_diagonal" : 0.1,
                               "off_diagonal" : 1
                              }
              ):
    on_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_on_diagonal_predicted, batch_labels["h_irreps_on_diagonal"]),
                    where=batch_labels["mask"])

    on_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_on_diagonal_predicted - batch_labels["h_irreps_on_diagonal"]),
                        where=batch_labels["mask"])

    off_diagonal_loss = jnp.mean(optax.huber_loss(h_irreps_off_diagonal_predicted, batch_labels["h_irreps_off_diagonal"]),
                    where=batch_labels["mask"])

    off_diagonal_mae_loss = jnp.mean(jnp.abs(h_irreps_off_diagonal_predicted - batch_labels["h_irreps_off_diagonal"]),
                        where=batch_labels["mask"])
    
    weighted_mean_loss = (loss_weights['on_diagonal']*on_diagonal_loss + \
                          loss_weights['off_diagonal']*off_diagonal_loss )/2
    weighted_mean_mae_loss = (loss_weights['on_diagonal']*on_diagonal_mae_loss + \
                              loss_weights['off_diagonal']*off_diagonal_mae_loss )/2
    return weighted_mean_loss, weighted_mean_mae_loss