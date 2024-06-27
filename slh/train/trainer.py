from time import time
from functools import partial
import logging
from typing import Union

log = logging.getLogger(__name__)

import flax.linen as nn
import jax.numpy as jnp
import jax
import optax

from tqdm import trange

from slh.data.input_pipeline import InMemoryDataset, PureInMemoryDataset
from slh.model.hmodel import HamiltonianModel
from slh.optimize.get_optimizer import get_opt


OptaxGradientTransformation = Union[optax.GradientTransformation]
OptimizerState = Union[optax.OptState, optax.MultiStepsState]



# def make_loss_function(params, batch)


@partial(jax.jit, static_argnames=("model_apply", "optimizer"))
def train_step(
    params: optax.Params, model_apply: callable, optimizer, batch_full_list: list, opt_state: OptimizerState
):
    step_mae_loss = 0.0
    for batch_full in batch_full_list:
        batch, batch_labels = batch_full

        def loss_function(params):
            h_irreps_predicted = model_apply(
                params,
                batch["numbers"],
                batch["idx_ij"],
                batch["idx_D"],
            )

            assert (
                h_irreps_predicted.shape
                == batch_labels["h_irreps"].shape
                == batch_labels["mask"].shape
            ), "This happens when your readout and your labels are not consistent."

            loss = jnp.mean(
                optax.huber_loss(h_irreps_predicted, batch_labels["h_irreps"]),
                where=batch_labels["mask"]
            )

            return loss, jnp.mean(
                jnp.abs(h_irreps_predicted - batch_labels["h_irreps"]),
                where=batch_labels["mask"],
            )

        (loss, mae_loss), grad = jax.value_and_grad(loss_function, has_aux=True)(params)
        updates, opt_state = optimizer.update(grad, opt_state)

        params = optax.apply_updates(params, updates)
        step_mae_loss += mae_loss

    return params, opt_state, grad, loss, step_mae_loss

@partial(jax.jit, static_argnames=("model_apply",))
def val_step(
    params, model_apply, batch_full_list
):
    step_mae_loss = 0.0
    for batch_full in batch_full_list:
        batch, batch_labels = batch_full

        def loss_function(params):
            h_irreps_predicted = model_apply(
                params,
                batch["numbers"],
                batch["idx_ij"],
                batch["idx_D"],
            )

            assert (
                h_irreps_predicted.shape
                == batch_labels["h_irreps"].shape
                == batch_labels["mask"].shape
            ), "This happens when your readout and your labels are not consistent."

            loss = jnp.mean(
                optax.huber_loss(h_irreps_predicted, batch_labels["h_irreps"]),
                where=batch_labels["mask"],
            )

            return loss, jnp.mean(
                jnp.abs(h_irreps_predicted - batch_labels["h_irreps"]),
                where=batch_labels["mask"],
            )

        loss, mae_loss = loss_function(params)
        step_mae_loss += mae_loss

    return loss, step_mae_loss


def fit(
    model: HamiltonianModel,
    train_dataset: PureInMemoryDataset,
    val_dataset: PureInMemoryDataset,
    n_grad_acc: int,
    n_epochs: int,
):

    # TODO: The val step
    train_batches_per_epoch = train_dataset.steps_per_epoch()
    val_batches_per_epoch = val_dataset.steps_per_epoch()
    # train_dataset = iter(train_dataset)
    batch_train_dataset = train_dataset.shuffle_and_batch()
    batch_val_dataset = val_dataset.shuffle_and_batch()

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

    cosine_lr = [
        dict(
            init_value=1e-2,
            peak_value=1e-1,
            warmup_steps=int(5 * tscale),
            decay_steps=int(30 * tscale),
            end_value=1e-3,
        )
        for tscale in jnp.linspace(1, 10, 10)
    ]

    optimizer = optax.adam(
        learning_rate=optax.exponential_decay(
        init_value=1e-2,
        transition_steps=1,
        decay_rate=0.99,
        transition_begin=20,
        staircase=False,
        end_value=1e-4),
        nesterov=True)
    # optimizer = optax.adamax(
    #     learning_rate=optax.exponential_decay(
    #         init_value=1e-2,
    #         transition_steps=5,
    #         decay_rate=0.9,
    #         transition_begin=50,
    #         staircase=False,
    #         end_value=1e-4,
    #     )
    # )

    optimizer = optax.MultiSteps(optimizer, every_k_schedule=n_grad_acc)

    # optimizer = optax.chain(
    #     optax.scale_by_adamax(),
    #     # optax.scale_by_learning_rate(learning_rate=optax.warmup_cosine_decay_schedule(
    #     # init_value=1e-4,
    #     # peak_value=1e-3,
    #     # warmup_steps=10,
    #     # decay_steps=50,
    #     # end_value=1e-4,
    #     # exponent=2.0
    #     # )),
    #     optax.scale_by_learning_rate(
    #         learning_rate=optax.sgdr_schedule(cosine_lr)
    #         ),
    #     # optax.clip(1.0)
    #     )

    # optimizer = get_opt(
    #     params,
    #     10,
    #     500,
    #     embedding_lr=0.01,
    #     ac_tensor_lr=0.005,
    #     bc_tensor_lr=0.005,
    #     dense_lr=0.001,
    #     exp_a_lr=0.01,
    #     exp_b_lr=0.001,
    #     exp_c_lr=0.001,
    #     default_lr=0.001,
    # )

    opt_state = optimizer.init(params)
    train_mae_loss = jnp.inf
    val_mae_loss = jnp.inf
    epoch_pbar = trange(n_epochs, desc="Epochs", ncols=75, disable=False, leave=True)

    try:
        mae_hist = []
        # lr_hist = []
        for epoch in range(n_epochs):
            effective_batch_size = n_grad_acc * train_dataset.batch_size

            train_batch_pbar = trange(0,
                train_batches_per_epoch // n_grad_acc,
                desc="Train batch",
                leave=False,
                ncols=75,
                smoothing=0.0,
                mininterval=1.0,
                disable=True,
            )

            epoch_mae_loss = 0.0
            for train_batch in range(0, train_batches_per_epoch // n_grad_acc):
                batch_data_list = [next(batch_train_dataset) for _ in range(n_grad_acc)]
                params, opt_state, _, loss, train_mae_loss = train_step(
                    params,
                    model_apply,
                    optimizer,
                    batch_data_list,
                    opt_state,
                )
                epoch_mae_loss += train_mae_loss

                train_batch_pbar.set_postfix(
                    mae=f"{train_mae_loss / effective_batch_size:0.3e}"
                )
                train_batch_pbar.update()

            val_batch_pbar = trange(0,
                val_batches_per_epoch // n_grad_acc,
                desc="Val batch",
                leave=False,
                ncols=75,
                smoothing=0.0,
                mininterval=1.0,
                disable=True,
            )

            epoch_mae_val_loss = 0.0
            for val_batch in range(0, val_batches_per_epoch // n_grad_acc):
                batch_data_list = [next(batch_val_dataset) for _ in range(n_grad_acc)]
                loss, val_mae_loss = val_step(
                    params,
                    model_apply,
                    batch_data_list,
                )
                epoch_mae_val_loss += val_mae_loss

                val_batch_pbar.set_postfix(
                    mae=f"{val_mae_loss / effective_batch_size:0.3e}"
                )
                val_batch_pbar.update()

            epoch_pbar.set_postfix(
                mae=f"{epoch_mae_val_loss / val_batches_per_epoch:0.3e}"
            )
            mae_hist.append(epoch_mae_val_loss / val_batches_per_epoch)
            epoch_pbar.update()
    
    except StopIteration or KeyboardInterrupt:
        print("Yes the stopiteration or the keyboard interrupt")
        return model, params
    
    import numpy as np
    mae_hist = np.array(mae_hist)
    np.savetxt("mae.log",mae_hist)

    return model, params
