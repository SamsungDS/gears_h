import logging
import time
from functools import partial
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import optax
from clu import metrics
from flax.training.train_state import TrainState
from tensorflow.keras.callbacks import CallbackList
from tqdm import trange

from slh.data.input_pipeline import PureInMemoryDataset

OptaxGradientTransformation = Union[optax.GradientTransformation]
OptimizerState = Union[optax.OptState, optax.MultiStepsState]

def fit(state: TrainState,
        train_dataset: PureInMemoryDataset,
        val_dataset: PureInMemoryDataset,
        logging_metrics: metrics.Collection,
        callbacks: CallbackList,
        n_grad_acc: int,
        n_epochs: int,
        ckpt_dir: Path,
        ckpt_interval: int = 1,
        is_ensemble: bool = False,
        data_parallel: bool = False,):
    
    return

def huber_loss(h_irreps_predicted, batch_labels):
    loss = jnp.mean(optax.huber_loss(h_irreps_predicted, batch_labels["h_irreps"]),
                    where=batch_labels["mask"])

    mae_loss = jnp.mean(jnp.abs(h_irreps_predicted - batch_labels["h_irreps"]),
                        where=batch_labels["mask"])
    return loss, mae_loss

def calculate_loss(params, batch_full, loss_function, model):
    batch, batch_labels = batch_full
    model_apply = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))
    # TODO Make loss function an argument and allow user input.
    h_irreps_predicted = model_apply(
        params,
        batch["numbers"],
        batch["idx_ij"],
        batch["idx_D"],
    )

    # TODO Remove this when we make the readout layer size automatically calculated.
    assert (
        h_irreps_predicted.shape
        == batch_labels["h_irreps"].shape
        == batch_labels["mask"].shape
    ), "This happens when your readout and your labels are not consistent."

    loss, mae_loss = loss_function(h_irreps_predicted, batch_labels)

    return loss, mae_loss

def make_step_functions(logging_metrics, model, loss_function = huber_loss):
    loss_calculator = partial(calculate_loss, loss_function=loss_function, model=model)
    grad_fn = jax.value_and_grad(loss_calculator, 0, has_aux=True)

    def update_step(state, batch_full):
        (loss, mae_loss), grads = grad_fn(state.params, batch_full)
        state = state.apply_gradients(grads=grads)
        return loss, mae_loss, state
    
    # TODO add support for ensemble models.

    @partial(jax.jit, static_argnames=("model_apply", "optimizer"))
    def train_step(state, batch):

        loss, mae_loss, state = update_step(state, batch)
        return loss, mae_loss, state
    
    # TODO OLD KEPT JUST IN CASE. Remove when no longer needed.
    # @partial(jax.jit, static_argnames=("model_apply", "optimizer"))
    # def train_step(params: optax.Params,
    #                model_apply: callable,
    #                optimizer,
    #                batch_full_list: list,
    #                opt_state: OptimizerState,):
        
    #     return params, opt_state, grad, loss, step_mae_loss
    
    
    @partial(jax.jit, static_argnames=("model_apply",))
    def val_step(state, batch):

        loss, mae_loss = loss_calculator(state, batch)

        return loss, mae_loss
    
    return train_step, val_step
