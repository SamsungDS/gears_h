from time import time
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jax
import optax


from surrogatelcaohamiltonians.data.input_pipeline import InMemoryDataset

# @partial(jax.jit, static_argnames=('model_apply', 'optimizer_update'))
def train_step(params, model_apply, optimizer_update, batch_full, opt_state):
  batch, batch_labels = batch_full
  def loss_function(params):
    h_irreps_predicted = model_apply(
      params,
      atomic_numbers=batch["numbers"], 
      neighbour_indices=batch["idx_ij"], 
      neighbour_displacements=batch["idx_D"])

    loss = jnp.mean( (h_irreps_predicted - batch_labels["h_irreps"]) ** 2, where=batch_labels["mask"])
    return loss, h_irreps_predicted
  
  (loss, h_irreps_predicted), grad = jax.value_and_grad(loss_function, has_aux=True)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss
  
  


def fit(model: nn.Module, 
        train_dataset: InMemoryDataset,
        # loss_function,
        n_epochs,
        ):
  # TODO: The val step
  steps_per_epoch = 5 # train_dataset.steps_per_epoch()
  train_dataset = iter(train_dataset)

  _batch_inputs, _batch_labels = next(train_dataset)
  params = model.init(jax.random.PRNGKey(2462), _batch_inputs["numbers"], _batch_inputs["idx_ij"], _batch_inputs["idx_D"])
  print(model.tabulate(jax.random.PRNGKey(2462), _batch_inputs["numbers"], _batch_inputs["idx_ij"], _batch_inputs["idx_D"]))
  optimizer = optax.adam(learning_rate=1e-3)
  opt_state = optimizer.init(params)


  for iepoch in range(1, n_epochs + 1):
    print("Epoch:", iepoch)
    for ibatch in range(steps_per_epoch):
      tstart = time()
      params, opt_state, loss = train_step(params, model.apply, optimizer.update, next(train_dataset), opt_state)
      print("Batch:", ibatch, "in", time() - tstart, "seconds. Loss:", loss)