from typing import Any
from functools import partial

import e3x
import jax
import jax.numpy as jnp
import jaxtyping

import flax.linen as nn

Array = jaxtyping.Array
Float = jaxtyping.Float
e3x.Config.set_cartesian_order(False)


class SpeciesAwareRadialBasis(nn.Module):
  cutoff: float
  num_radial: int = 8
  max_degree: int = 3
  num_elemental_embedding: int = 64

  def setup(self):
    self.radial_function = partial(e3x.nn.sinc, limit=self.cutoff)
    # TODO Do we really want anything more than Bismuth? No. No, we do not.
    self.embedding = e3x.nn.Embed(83, self.num_elemental_embedding, name="elemental embedding")
  
  @nn.compact
  def __call__(self, neighbour_displacements, Z_j):
    """_summary_

    Parameters
    ----------
    neighbour_displacements : _type_
        _description_
    Z_j : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    basis_expansion = e3x.nn.basis(
      neighbour_displacements,
      num=self.num_radial,
      max_degree=self.max_degree,
      radial_fn=self.radial_function,
      cutoff_fn=partial(e3x.nn.cosine_cutoff, cutoff=self.cutoff),
      )
    
    # We transform the embedding dimension to the radial basis dimension 
    # so we can product meaningfully
    transformed_embedding = e3x.nn.Dense(self.num_radial, name="transform embedding")(self.embedding(Z_j))
    y = e3x.nn.Tensor(max_degree=self.max_degree, include_pseudotensors=False, name="emb x basis")(transformed_embedding, basis_expansion)
    # TODO This swish here is just for nonlinearity. I don't know if we actually need it.
    return e3x.nn.swish(y)


def jinclike(x: Float[Array, '...'], num: int, limit: float = 1.0):
  r"""Jinc-like functions from https://arxiv.org/pdf/1907.02374.pdf

  We use the f_n version of the functions, without orthogonalizing. 
  The g_n version of these functions, which are orthogonal, 
  require a recurrence relation to be calculated, which is perhaps 
  not the play at least for the moment, especially given how 
  absolutely tiny the differences between them are.

  Parameters
  ----------
  x : Float[Array, &#39;...&#39;]
      Input array
  num : int
      Number of basis functions
  limit : float, optional
      Basis functions are distributed between 0 and ``limit``, by default 1.0

  Returns
  -------
  floating array
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.

  Raises
  ------
  ValueError
      _description_
  """
  if num < 1:
    raise ValueError(f'num must be greater or equal to 1, received {num}')
  
  with jax.ensure_compile_time_eval():
    i = jnp.arange(0, num)
    factor1 = 2.0 ** 0.5 * jnp.pi / limit ** 1.5
    factor2 = (-1) ** i * (i + 1) * (i + 2) / jnp.hypot(i + 1, i + 2)
  
  x = jnp.expand_dims(x, axis=-1)

  return factor1 * factor2 * \
    (jnp.sinc(x / limit * (i + 1)) + jnp.sinc(x / limit * (i + 2)))



