# Large parts of this file are from https://github.com/apax-hub/apax/blob/dev/apax/train/checkpoints.py
import logging

import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from pathlib import Path

log = logging.getLogger(__name__)


def check_for_ensemble(params: FrozenDict) -> int:    """Checks if a set of parameters belongs to an ensemble model.
    This is the case if all parameters share the same first dimension (parameter batch)
    """
    flat_params = flatten_dict(params)
    shapes = [v.shape[0] for v in flat_params.values()]
    is_ensemble = len(set(shapes)) == 1

    if is_ensemble:
        return shapes[0]
    else:
        return 1

def create_train_state(model, params: FrozenDict, tx):
    n_models = check_for_ensemble(params)

    def create_single_train_state(params):
        state = train_state.TrainState.create(
            apply_fn=model,
            params=params,
            tx=tx,
        )
        return state

    if n_models > 1:
        train_state_fn = jax.vmap(create_single_train_state, axis_name="ensemble")
    else:
        train_state_fn = create_single_train_state

    return train_state_fn(params)

def create_params(model, rng_key, sample_input: tuple, n_models: int):
    keys = jax.random.split(rng_key, num=n_models + 1)
    rng_key, model_rng = keys[0], keys[1:]

    log.info(f"initializing {n_models} models")

    if n_models == 1:
        params = model.init(model_rng[0], *sample_input)
    elif n_models > 1:
        num_args = len(sample_input)
        # vmap only over parameters, not over any data from the input
        in_axes = (0, *[None] * num_args)
        params = jax.vmap(model.init, in_axes=in_axes)(model_rng, *sample_input)
    else:
        raise ValueError(f"n_models should be a positive integer, found {n_models}")

    params = freeze(params)

    return params, rng_key

class CheckpointManager:
    def __init__(self) -> None:
        self.async_manager = checkpoints.AsyncManager()

    def save_checkpoint(self, ckpt, epoch: int, path: Path) -> None:
        checkpoints.save_checkpoint(
            ckpt_dir=path.resolve(),
            target=ckpt,
            step=epoch,
            overwrite=True,
            keep=2,
            async_manager=self.async_manager,
        )