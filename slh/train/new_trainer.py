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

def make_step_functions(loss_function, logging_metrics, model):

    @partial(jax.jit, static_argnames=("model_apply", "optimizer"))
    def train_step(params: optax.Params,
                   model_apply: callable,
                   optimizer,
                   batch_full_list: list,
                   opt_state: OptimizerState,):
        
        return params, opt_state, grad, loss, step_mae_loss
    
    @partial(jax.jit, static_argnames=("model_apply",))
    def val_step(params, 
                 model_apply, 
                 batch_full_list):

        return loss, step_mae_loss
    
    return train_step, val_step
