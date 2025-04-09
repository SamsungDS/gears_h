# Based on https://github.com/apax-hub/apax/blob/dev/apax/train/checkpoints.py
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any, List

import jax
import jax.numpy as jnp
from flax import core, struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

log = logging.getLogger(__name__)


def check_for_ensemble(params: FrozenDict) -> int:
    """Checks if a set of parameters belongs to an ensemble model.
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
        
        state = TrainStateExtraArgs.create(
            apply_fn=model,
            params=params,
            tx=tx,  # This is an optimizer state
        )
        return state


    if n_models > 1:
        train_state_fn = jax.vmap(create_single_train_state, axis_name="ensemble")
    else:
        train_state_fn = create_single_train_state

    return train_state_fn(params)


def create_params(model, rng_key, sample_input: dict, n_models: int):
    keys = jax.random.split(rng_key, num=n_models + 1)
    rng_key, model_rng = keys[0], keys[1:]

    log.info(f"initializing {n_models} models")

    if n_models == 1:
        params = model.init(
            model_rng[0],
            jnp.array(sample_input["numbers"]),
            jnp.array(sample_input["idx_ij"]),
            jnp.array(sample_input["idx_D"]),
            jnp.array(sample_input["idx_bonds"])
        )
        # params = model.init(jax.random.PRNGKey(245), *sample_input)
    elif n_models > 1:
        raise NotImplementedError
        # num_args = len(sample_input)
        # # vmap only over parameters, not over any data from the input
        # in_axes = (0, *[None] * num_args)
        # params = jax.vmap(model.init, in_axes=in_axes)(model_rng, *sample_input)
    else:
        raise ValueError(f"n_models should be a positive integer, found {n_models}")

    params = freeze(params)

    return params, rng_key


def load_state(state, ckpt_dir):
    start_epoch = 0
    target = {"model": state, "epoch": 0}
    checkpoints_exist = Path(ckpt_dir).is_dir()
    if checkpoints_exist:
        log.info("Loading checkpoint")
        raw_restored = checkpoints.restore_checkpoint(
            ckpt_dir, target=target, step=None
        )
        state = raw_restored["model"]
        start_epoch = raw_restored["epoch"] + 1
        log.info(
            "Successfully restored checkpoint from epoch %d", raw_restored["epoch"]
        )

    return state, start_epoch


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


def stack_parameters(param_list: List[FrozenDict]) -> FrozenDict:
    """Combine a list of parameter sets into a stacked version.
    Used for model ensembles.
    """
    flat_param_list = []
    for params in param_list:
        params = unfreeze(params)
        flat_params = flatten_dict(params)
        flat_param_list.append(flat_params)

    stacked_flat_params = flat_params
    for p in flat_param_list[0].keys():
        stacked_flat_params[p] = jnp.stack(
            [flat_param[p] for flat_param in flat_param_list]
        )

    stacked_params = unflatten_dict(stacked_flat_params)
    stack_params = freeze(stacked_params)
    return stack_params


def load_params(model_version_path: Path, best=True) -> FrozenDict:
    model_version_path = Path(model_version_path)
    if best:
        model_version_path = model_version_path / "best"
    log.info(f"loading checkpoint from {model_version_path}")
    try:
        # keep try except block for zntrack load from rev
        raw_restored = checkpoints.restore_checkpoint(
            model_version_path, target=None, step=None
        )
    except FileNotFoundError:
        print(f"No checkpoint found at {model_version_path}")
    if raw_restored is None:
        raise FileNotFoundError(f"No checkpoint found at {model_version_path}")
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params


# def restore_single_parameters(model_dir: Path) -> Tuple[Config, FrozenDict]:
#     """Load the config and parameters of a single model"""
#     model_dir = Path(model_dir)
#     model_config = parse_config(model_dir / "config.yaml")

#     if model_config.data.experiment == "":
#         model_config.data.directory = model_dir.resolve().as_posix()
#     else:
#         model_config.data.directory = model_dir.parent.resolve().as_posix()

#     ckpt_dir = model_config.data.model_version_path
#     return model_config, load_params(ckpt_dir)

class TrainStateExtraArgs(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.
  Very small modification of the TrainState shipped with flax.

  Example usage::

    >>> import flax.linen as nn
    >>> from flax.training.train_state import TrainState
    >>> import jax, jax.numpy as jnp
    >>> import optax

    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 2))
    >>> model = nn.Dense(2)
    >>> variables = model.init(jax.random.key(0), x)
    >>> tx = optax.adam(1e-3)

    >>> state = TrainState.create(
    ...     apply_fn=model.apply,
    ...     params=variables['params'],
    ...     tx=tx)

    >>> def loss_fn(params, x, y):
    ...   predictions = state.apply_fn({'params': params}, x)
    ...   loss = optax.l2_loss(predictions=predictions, targets=y).mean()
    ...   return loss
    >>> loss_fn(state.params, x, y)
    Array(1.8136346, dtype=float32)

    >>> grads = jax.grad(loss_fn)(state.params, x, y)
    >>> state = state.apply_gradients(grads=grads)
    >>> loss_fn(state.params, x, y)
    Array(1.8079796, dtype=float32)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      ``.apply_gradients()``.
    apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for
      convenience to have a shorter params list for the ``train_step()`` function
      in your training loop.
    params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
    tx: An Optax gradient transformation.
    opt_state: The state for ``tx``.
  """

  step: int | jax.Array
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformationExtraArgs = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, value, **kwargs):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: Gradients that have the same pytree structure as ``.params``.
      **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

    Returns:
      An updated instance of ``self`` with ``step`` incremented by one, ``params``
      and ``opt_state`` updated by applying ``grads``, and additional attributes
      replaced as specified by ``kwargs``.
    """
    if OVERWRITE_WITH_GRADIENT in grads:
      grads_with_opt = grads['params']
      params_with_opt = self.params['params']
    else:
      grads_with_opt = grads
      params_with_opt = self.params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt, value=value
    )
    new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    # As implied by the OWG name, the gradients are used directly to update the
    # parameters.
    if OVERWRITE_WITH_GRADIENT in grads:
      new_params = {
        'params': new_params_with_opt,
        OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
      }
    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
    # We exclude OWG params when present because they do not need opt states.
    params_with_opt = (
      params['params'] if OVERWRITE_WITH_GRADIENT in params else params
    )
    opt_state = tx.init(params_with_opt)
    return cls(
      step=0,
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=opt_state,
      **kwargs,
    )
