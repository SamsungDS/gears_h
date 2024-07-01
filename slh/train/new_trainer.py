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
from slh.train.checkpoints import load_state
from slh.train.loss import huber_loss

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
        data_parallel: bool = False,
        loss_function = huber_loss):
    
    # Error handling here
    if data_parallel:
        raise NotImplementedError
    
    # Checkpointing directories and manager
    latest_dir = ckpt_dir / "latest"
    best_dir = ckpt_dir / "best"
    ckpt_manager = CheckpointManager()

    # Dataset batching and shuffling
    train_batches_per_epoch = train_dataset.steps_per_epoch()
    val_batches_per_epoch = val_dataset.steps_per_epoch()
    batch_train_dataset = train_dataset.shuffle_and_batch()
    batch_val_dataset = val_dataset.shuffle_and_batch()

    # Create train_step and val_step functions
    train_step, val_step = make_step_functions(logging_metrics,
                                               model = state.apply_fn,
                                               loss_function = loss_function
                                               )
    
    state, start_epoch = load_state(state, latest_dir)
    if start_epoch >= n_epochs:
        raise ValueError(
            f"n_epochs <= current epoch from checkpoint ({n_epochs} <= {start_epoch})"
        )
    
    best_params = {} # TODO do we need this if we're saving the state?
    best_mae_loss = jnp.inf
    epoch_loss = {}

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()

        effective_batch_size = n_grad_acc * train_dataset.batch_size

        # Training set loop
        epoch_loss.update({"train_loss": 0.0})
        train_mae_loss = jnp.inf

        train_batch_pbar = trange(
            0,
            train_batches_per_epoch
            // n_grad_acc,  # TODO this is a big fragile if train_batches_per_epoch isn't a multiple
            desc="Train batch",
            leave=False,
            ncols=75,
            smoothing=0.0,
            mininterval=1.0,
            disable=True,
        )
        for train_batch in range(train_batches_per_epoch // n_grad_acc):
            batch_data_list = [next(batch_train_dataset) for _ in range(n_grad_acc)]
            loss, mae_loss, state = train_step(state, batch_data_list)
            
            train_mae_loss += mae_loss
            epoch_loss["train_loss"] += loss

            train_batch_pbar.set_postfix(
                mae=f"{train_mae_loss / effective_batch_size:0.3e}"
            )
            train_batch_pbar.update()

    return

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
