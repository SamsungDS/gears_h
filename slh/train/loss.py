import jax.numpy as jnp
import optax


def huber_loss(h_irreps_predicted, batch_labels):
    loss = jnp.mean(optax.huber_loss(h_irreps_predicted, batch_labels["h_irreps"]),
                    where=batch_labels["mask"])

    mae_loss = jnp.mean(jnp.abs(h_irreps_predicted - batch_labels["h_irreps"]),
                        where=batch_labels["mask"])
    return loss, mae_loss