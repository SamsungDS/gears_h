from time import time
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jax
import optax

from tqdm import trange

from surrogatelcaohamiltonians.data.input_pipeline import InMemoryDataset, PureInMemoryDataset
from surrogatelcaohamiltonians.model.hmodel import HamiltonianModel


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

        loss = jnp.mean(
            (h_irreps_predicted - batch_labels["h_irreps"]) ** 2,
            where=batch_labels["mask"],
        )
        return loss, h_irreps_predicted

    (loss, h_irreps_predicted), grad = jax.value_and_grad(loss_function, has_aux=True)(
        params
    )
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


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
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    for iepoch in range(n_epochs):
        print("Epoch:", iepoch)
        for ibatch in trange(steps_per_epoch):
            tstart = time()
            params, opt_state, loss = train_step(
                params, model_apply, optimizer.update, next(batch_train_dataset), opt_state
            )
            # jax.block_until_ready(loss)
            # print("Batch:", ibatch, "in", time() - tstart, "seconds. Loss:", loss)
        print("Loss:", loss)

    # train_dataset.cleanup()