import logging
import time
from functools import partial
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
from clu import metrics
from flax.training.train_state import TrainState
# from orbax.checkpointing import CheckpointManager, CheckpointManagerOptions # for potential orbax migration
from tensorflow.keras.callbacks import CallbackList
from tqdm import trange

from slh.data.input_pipeline import PureInMemoryDataset
from slh.train.checkpoints import CheckpointManager, load_state
from slh.train.loss import huber_loss

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
        loss_function = huber_loss,
        disable_pbar: bool = False,
        disable_batch_pbar: bool = True):
    
    # Error handling here
    # TODO implement these functionalities.
    if data_parallel:
        raise NotImplementedError
    if is_ensemble:
        raise NotImplementedError
    if n_grad_acc > 1:
        raise NotImplementedError

    log.info("Beginning Training")
    callbacks.on_train_begin()
    
    # Checkpointing directories and manager
    latest_dir = ckpt_dir / "latest"
    best_dir = ckpt_dir / "best"
    # TODO Migrate to orbax checkpointing (?). Currently uses the legacy flax checkpointing API.
    ckpt_manager = CheckpointManager()
    # For potential orbax migration. inelegant but currently the easiest way to handle both best and latest.
    # latest_ckpt_manager_options = CheckpointManagerOptions(max_to_keep=100, save_interval_steps=1)
    # latest_ckpt_manager = CheckpointManager(path = latest_dir, options = latest_ckpt_manager_options)
    # best_ckpt_manager_options = CheckpointManagerOptions(max_to_keep=5, save_interval_steps=1)
    # best_ckpt_manager = CheckpointManager(path = best_dir, options = best_ckpt_manager_options)

    # Dataset batching and shuffling
    train_batches_per_epoch = train_dataset.steps_per_epoch()
    val_batches_per_epoch = val_dataset.steps_per_epoch()
    batch_train_dataset = train_dataset.shuffle_and_batch()
    batch_val_dataset = val_dataset.shuffle_and_batch()

    # Create train_step and val_step functions
    train_step, val_step = make_step_functions(logging_metrics,
                                               state = state,
                                               loss_function = loss_function
                                               )
    
    state, start_epoch = load_state(state, latest_dir)
    if start_epoch >= n_epochs:
        raise ValueError(
            f"n_epochs <= current epoch from checkpoint ({n_epochs} <= {start_epoch})"
        )

    best_params = {} # TODO do we need this if we're saving the state?
    best_mae_loss = jnp.inf
    best_loss = float(jnp.inf)
    epoch_loss = {}

    epoch_pbar = trange(
        start_epoch, n_epochs, desc="Epochs", ncols=100, disable=disable_pbar, leave=True
    )
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch=epoch + 1)

        effective_batch_size = n_grad_acc * train_dataset.batch_size

        # Training set loop - set up
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
            disable=disable_batch_pbar,
        )
        # Training set loop - actual training
        for train_batch in range(train_batches_per_epoch // n_grad_acc):
            callbacks.on_train_batch_begin(batch=train_batch)
            batch_data_list = [next(batch_train_dataset) for _ in range(n_grad_acc)]
            # TODO refactor train_step for gradient accumulation and remove the hardcoded first element of the list below.
            loss, mae_loss, state = train_step(state, batch_data_list[0])
            
            train_mae_loss += mae_loss
            epoch_loss["train_loss"] += loss

            train_batch_pbar.set_postfix(
                mae=f"{train_mae_loss / effective_batch_size:0.3e}"
            )
            callbacks.on_train_batch_end(batch=train_batch)
            train_batch_pbar.update()
        
        epoch_loss["train_loss"] /= train_batches_per_epoch
        epoch_loss["train_loss"] = float(epoch_loss["train_loss"])


        # Validation set loop - set up
        epoch_loss.update({"val_loss": 0.0})
        epoch_val_mae_accumulator = 0.0

        val_batch_pbar = trange(
            0,
            val_batches_per_epoch // n_grad_acc, # TODO is this fragile like the training loop equivalent?
            desc="Val batch",
            leave=False,
            ncols=75,
            smoothing=0.0,
            mininterval=1.0,
            disable=disable_batch_pbar,
        )
        # Validation set loop - actual training
        for val_batch in range(val_batches_per_epoch // n_grad_acc):
            batch_data_list = [next(batch_val_dataset) for _ in range(n_grad_acc)]
            # TODO refactor train_step for gradient accumulation and remove the hardcoded first element of the list below.
            loss, mae_loss = val_step(state, batch_data_list[0])

            epoch_val_mae_accumulator += mae_loss
            epoch_loss["val_loss"] += loss

            val_batch_pbar.set_postfix(
                mae=f"{epoch_val_mae_accumulator / val_batches_per_epoch:0.3e}"
            )
        
        epoch_loss["val_loss"] /= val_batches_per_epoch
        epoch_loss["val_loss"] = float(epoch_loss["val_loss"])

        if (epoch_val_mae_accumulator / val_batches_per_epoch) < best_mae_loss:
                best_mae_loss = epoch_val_mae_accumulator / val_batches_per_epoch
                best_params['params'] = state.params

        epoch_end_time = time.time()
        # TODO store this elsewhere?
        epoch_loss['epoch_time'] = epoch_end_time - epoch_start_time
        callbacks.on_epoch_end(epoch=epoch, logs=epoch_loss)

        ckpt = {"model": state, "epoch": epoch}
        if epoch % ckpt_interval == 0:
            ckpt_manager.save_checkpoint(ckpt, epoch, latest_dir)

        if epoch_loss["val_loss"] < best_loss:
            best_loss = epoch_loss["val_loss"]
            ckpt_manager.save_checkpoint(ckpt, epoch, best_dir)

        epoch_pbar.set_postfix(mae=f"{epoch_val_mae_accumulator / val_batches_per_epoch:0.3e}")
        epoch_pbar.update()
    epoch_pbar.close()
    callbacks.on_train_end()


    return

def calculate_loss(params, batch_full, loss_function, apply_function):
    batch, batch_labels = batch_full
    model_apply = apply_function#jax.vmap(apply_function, in_axes=(None, 0, 0, 0))
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

def make_step_functions(logging_metrics, state, loss_function = huber_loss):
    loss_calculator = partial(calculate_loss, loss_function=loss_function, apply_function=state.apply_fn)
    grad_fn = jax.value_and_grad(loss_calculator, 0, has_aux=True)

    def update_step(state, batch_full):
        (loss, mae_loss), grads = grad_fn(state.params, batch_full)
        state = state.apply_gradients(grads=grads)
        return loss, mae_loss, state
    
    # TODO add support for ensemble models.

    # @partial(jax.jit, static_argnames=("model_apply", "optimizer"))
    @jax.jit
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
    
    
    # @partial(jax.jit, static_argnames=("model_apply",))
    @jax.jit
    def val_step(state, batch):

        loss, mae_loss = loss_calculator(state.params, batch)

        return loss, mae_loss
    
    return train_step, val_step
