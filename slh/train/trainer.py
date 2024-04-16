from time import time
from functools import partial
import logging

log = logging.getLogger(__name__)

import flax.linen as nn
import jax.numpy as jnp
import jax
import optax

from tqdm import trange

from slh.data.input_pipeline import InMemoryDataset, PureInMemoryDataset
from slh.model.hmodel import HamiltonianModel


@partial(jax.jit, static_argnames=("model_apply", "optimizer_update"))
def train_step(params, model_apply, optimizer_update, batch_full, opt_state):
    batch, batch_labels = batch_full

    def loss_function(params):
        h_irreps_predicted = model_apply(
            params,
            batch["numbers"],
            batch["idx_ij"],
            batch["idx_D"],
        )

        # loss = jnp.mean(
        #     (h_irreps_predicted - batch_labels["h_irreps"]) ** 2,
        #     where=batch_labels["mask"],
        # )

        assert h_irreps_predicted.shape == batch_labels["h_irreps"].shape == batch_labels["mask"].shape, "This happens when your readout and your labels are not consistent."

        loss = jnp.mean(
            optax.huber_loss(h_irreps_predicted, batch_labels["h_irreps"]),
            where=batch_labels["mask"],
        )

        return loss, jnp.mean(
            jnp.abs(h_irreps_predicted - batch_labels["h_irreps"]),
            where=batch_labels["mask"],
        )

    (loss, mae_loss), grad = jax.value_and_grad(loss_function, has_aux=True)(params)
    # grad = optax.adaptive_grad_clip(0.1)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    # updates, opt_state = optax.chain()
    params = optax.apply_updates(params, updates)
    return params, opt_state, grad, mae_loss


def fit(
    model: HamiltonianModel,
    train_dataset: PureInMemoryDataset,
    # loss_function,
    n_epochs: int,
):
    # TODO: The val step
    steps_per_epoch = train_dataset.steps_per_epoch()
    # train_dataset = iter(train_dataset)
    batch_train_dataset = train_dataset.shuffle_and_batch()

    # We want to batch this over all inputs, but not the parameters of the model
    model_apply = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))

    _batch_inputs, _batch_labels = next(batch_train_dataset)
    # print(len(_batch_inputs["idx_ij"][0]))
    params = model.init(
        jax.random.PRNGKey(2462),
        atomic_numbers=_batch_inputs["numbers"][0],
        neighbour_indices=_batch_inputs["idx_ij"][0],
        neighbour_displacements=_batch_inputs["idx_D"][0],
    )
    print(
        model.tabulate(
            jax.random.PRNGKey(2462),
            _batch_inputs["numbers"][0],
            _batch_inputs["idx_ij"][0],
            _batch_inputs["idx_D"][0],
        )
    )

    # # model_apply(
    # #     _params,
    # #     _batch_inputs["numbers"],
    # #     _batch_inputs["idx_ij"],
    # #     _batch_inputs["idx_D"],
    # # )
    # optimizer = optax.adamaxw(learning_rate=optax.warmup_cosine_decay_schedule(1e-4, 5e-3, 5, 20, 1e-4))# , nesterov=True)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    mae_loss = jnp.nan
    epoch_pbar = trange(n_epochs, desc="Epochs", ncols=75, disable=False)

    params_list, grad_list = [], []
    try:
        for _ in epoch_pbar:
            batch_pbar = trange(steps_per_epoch, desc="Batch", leave=False, ncols=75, smoothing=0.0, disable=False)
            epoch_mae_loss = 0.0
            for _ in batch_pbar:
                batch_data = next(batch_train_dataset)
                # log.info(f"{batch_data[1]['h_irreps'].shape}")
                params, opt_state, grad, mae_loss = train_step(
                    params,
                    model_apply,
                    optimizer.update,
                    batch_data,
                    opt_state,
                )
                epoch_mae_loss += mae_loss
                # params_list.append(params)
                # grad_list.append(grad)

                batch_pbar.set_postfix(mae=f"{mae_loss / train_dataset.batch_size:0.3e}")
                batch_pbar.update()
            
            epoch_pbar.set_postfix(mae=f"{epoch_mae_loss / (steps_per_epoch  * train_dataset.batch_size):0.3e}")
            epoch_pbar.update()
    except StopIteration:
        print("Yes the stopiteration")
    
    return params_list, grad_list