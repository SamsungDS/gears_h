import jax.numpy as jnp
import jax
import optax

from surrogatelcaohamiltonians.data.input_pipeline import InMemoryDataset


def train_step(params, model_apply, optimizer_update, batch):
  def loss_function(params):
    h_irreps_predicted = model_apply(atomic_numbers=batch["numbers"], neighbour_indices=batch["idx_ij"], neighbour_displacements=batch=["idx_D"])

    loss = jnp.mean( ((h_irreps_predicted - batch["h_irreps"]) * batch["mask"]) ** 2)
    return loss, h_irreps_predicted
  
  (loss, h_irreps_predicted), grad = jax.value_and_grad(loss_function, has_aux=True)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss
  
  


def fit(model, 
        train_dataset: InMemoryDataset,
        loss_function,
        n_epochs,
        ):
  # TODO: The val step

  steps_per_epoch = train_dataset.steps_per_epoch()
  for epoch in range(1, n_epochs + 1):
    for ibatch in r