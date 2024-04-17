import logging
from typing import Any, Callable

import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze

log = logging.getLogger(__name__)


def map_nested_fn(fn: Callable[[str, Any], dict]) -> Callable[[dict], dict]:
    """
    Recursively apply `fn` to the key-value pairs of a nested dict
    See
    https://optax.readthedocs.io/en/latest/api.html?highlight=multitransform#multi-transform
    """

    def map_fn(nested_dict):
        return {
            k: map_fn(v) if isinstance(v, dict) else fn(k, v)
            for k, v in nested_dict.items()
        }

    return map_fn


def get_schedule(
    lr: float, transition_begin: int, transition_steps: int
) -> optax._src.base.Schedule:
    """
    builds a linear learning rate schedule.
    """
    lr_schedule = optax.linear_schedule(
        init_value=lr,
        end_value=1e-6,
        transition_begin=transition_begin,
        transition_steps=transition_steps,
    )
    return lr_schedule


def make_optimizer(opt, lr, transition_begin, transition_steps, opt_kwargs):
    if lr <= 1e-7:
        optimizer = optax.set_to_zero()
    else:
        schedule = get_schedule(lr, transition_begin, transition_steps)
        optimizer = opt(schedule, **opt_kwargs)
    return optimizer


def get_opt(
    params,
    transition_begin: int,
    transition_steps: int,
    embedding_lr: float = 0.02,
    ac_tensor_lr: float = 0.001,
    bc_tensor_lr: float = 0.001,
    dense_lr: float = 0.003,
    exp_a_lr: float = 0.01,
    exp_b_lr: float = 0.001,
    exp_c_lr: float = 0.001,
    default_lr: float = 0.001,
    opt_name: str = "adam",
    opt_kwargs: dict = {},
    **kwargs,
) -> optax._src.base.GradientTransformation:
    """
    Builds an optimizer with different learning rates for each parameter group.
    Several `optax` optimizers are supported.
    """
    log.info("Initializing Optimizer")
    opt = getattr(optax, opt_name)

    embedding_opt = make_optimizer(opt, embedding_lr, transition_begin, transition_steps, opt_kwargs)
    atomcentered_tensor_opt = make_optimizer(opt, ac_tensor_lr, transition_begin, transition_steps, opt_kwargs)
    bondcentered_tensor_opt = make_optimizer(opt, bc_tensor_lr, transition_begin, transition_steps, opt_kwargs)
    
    dense_opt = make_optimizer(opt, dense_lr, transition_begin, transition_steps, opt_kwargs)
    
    exp_a_opt = make_optimizer(
        opt, exp_a_lr, transition_begin, transition_steps, opt_kwargs
    )
    exp_b_opt = make_optimizer(
        opt, exp_b_lr, transition_begin, transition_steps, opt_kwargs
    )

    exp_c_opt = make_optimizer(
        opt, exp_c_lr, transition_begin, transition_steps, opt_kwargs
    )

    default_opt = make_optimizer(
        opt, default_lr, transition_begin, transition_steps, opt_kwargs
    )

    partition_optimizers = {
        "elem_embed": embedding_opt,
        "embed_transform": embedding_opt,
        "tensor_embed_basis": embedding_opt,
        "atom_centered": atomcentered_tensor_opt,
        "bond_centered": bondcentered_tensor_opt,
        "dense": dense_opt,
        "prefactors": exp_a_opt,
        "exponents": exp_b_opt,
        "offsets": exp_c_opt,
        "default": default_opt,
    }

    def subset_specialization(path: tuple[str], v):
        this_partition = 'default'

        for branch in path:
            if branch in partition_optimizers:
                this_partition = branch

        return this_partition

    param_partitions = traverse_util.path_aware_map(subset_specialization, params)
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return tx