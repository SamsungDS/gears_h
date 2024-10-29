import functools
import math
from typing import Any, Optional, Sequence, Union

from e3x import ops
from e3x.nn import initializers
from e3x.nn.features import _extract_max_degree_and_check_shape
from e3x.nn.features import change_max_degree_or_type
from e3x.nn.modules import _Conv, Dense, _duplication_indices_for_max_degree
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping

FusedTensorInitializerFn = initializers.FusedTensorInitializerFn
InitializerFn = initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Shape = Sequence[Union[int, Any]]
Dtype = Any  # This could be a real type if support for that is added.
PRNGKey = UInt32[Array, '2']
PrecisionLike = jax.lax.PrecisionLike

default_kernel_init = jax.nn.initializers.lecun_normal()

class TanhThrMultiHeadAttention(_Conv):
  r"""Equivariant multi-head attention with Tanh Thresholding added.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
    num_heads: Number of attention heads.
    qkv_features: Number of features used for queries, keys and values. If this
      is `None`, the same number of features as in `inputs_q` is used.
    out_features: Number of features for the output. If this is `None`, the same
      number of features as in `inputs_q` is used.
    use_relative_positional_encoding_qk: If this is `True`, relative positional
      encodings are used for computing the dot product between queries and keys.
    use_relative_positional_encoding_v: If this is `True`, a relative positional
      encoding (with respect to the queries) is used for computing the values.
    query_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing queries.
    query_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing queries.
    query_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing queries.
    key_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing keys.
    key_bias_init: Initializer function for the bias terms of the :class:`Dense`
      layer for computing keys.
    key_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing keys.
    value_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing values.
    value_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing values.
    value_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing values.
    output_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing outputs.
    output_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing outputs.
    output_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing outputs.
  """

  num_heads: Optional[int] = 1
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  use_relative_positional_encoding_qk: bool = True
  use_relative_positional_encoding_v: bool = True
  query_kernel_init: InitializerFn = default_kernel_init
  query_bias_init: InitializerFn = jax.nn.initializers.zeros
  query_use_bias: bool = False
  key_kernel_init: InitializerFn = default_kernel_init
  key_bias_init: InitializerFn = jax.nn.initializers.zeros
  key_use_bias: bool = False
  value_kernel_init: InitializerFn = default_kernel_init
  value_bias_init: InitializerFn = jax.nn.initializers.zeros
  value_use_bias: bool = False
  output_kernel_init: InitializerFn = default_kernel_init
  output_bias_init: InitializerFn = jax.nn.initializers.zeros
  output_use_bias: bool = True

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs_q: Union[
          Float[Array, '... N 1 (max_degree+1)**2 q_features'],
          Float[Array, '... N 2 (max_degree+1)**2 q_features'],
      ],
      inputs_kv: Union[
          Float[Array, '... M 1 (max_degree+1)**2 kv_features'],
          Float[Array, '... M 2 (max_degree+1)**2 kv_features'],
      ],
      basis: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      cutoff_value: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      src_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (max_degree+1)**2 out_features'],
      Float[Array, '... N 2 (max_degree+1)**2 out_features'],
  ]:
    """Applies multi-head attention.

    Args:
      inputs_q: Input features that are used to compute queries.
      inputs_kv: Input features that are used to compute keys and values.
      basis: Basis functions for all relevant interactions between queries and
        keys (either in dense or sparse indexed format).
      cutoff_value: Multiplicative cutoff values that are applied to the "raw"
        softmax values (before normalization), can be used for smooth cutoffs.
      adj_idx: Adjacency indices (dense index list), or `None`.
      where: Mask to specify which values to sum over (only for dense index
        lists). If this is `None`, the `where` mask is auto-determined from
        `inputs_kv`.
      dst_idx: Destination indices (sparse index list), or `None`.
      src_idx: Source indices (sparse index list), or `None`.
      num_segments: Number of segments after summation (only for sparse index
        lists). If this is `None`, `num_segments` is auto-determined from
        `inputs_q`.
      indices_are_sorted: If `True`, `dst_idx` is assumed to be sorted, which
        may increase performance (only used for sparse index lists).

    Returns:
      The result of the multi-head attention computation.

    Raises:
      ValueError: If `inputs_q` and `inputs_kv` have incompatible shapes, or if
        `qkv_features` is not divisible by `num_heads`.
      TypeError: When relative positional encodings are requested, but no input
        for `basis` is provided.
    """

    # Shape check.
    if inputs_q.shape[:-4] != inputs_kv.shape[:-4]:
      raise ValueError('inputs_q and inputs_kv have incompatible shapes')

    # Check that positional encodings are possible.
    if (
        self.use_relative_positional_encoding_qk
        or self.use_relative_positional_encoding_v
    ) and basis is None:
      raise TypeError(
          "when using relative positional encodings, 'basis' is "
          'a required argument, received basis=None'
      )

    # Determine features and check for compatibility with num_heads.
    out_features = (
        inputs_q.shape[-1] if self.out_features is None else self.out_features
    )
    qkv_features = (
        inputs_q.shape[-1] if self.qkv_features is None else self.qkv_features
    )

    if qkv_features % self.num_heads != 0:
      raise ValueError(
          f'qkv_features ({qkv_features}) must be divisible by '
          f'num_heads ({self.num_heads})'
      )

    # For query and key projections (used to calculate the dot product), we have
    # to make sure that the final query and key have the same number of
    # parity/degree channels, or the dot product would be ill-defined.
    max_degree_q = _extract_max_degree_and_check_shape(inputs_q.shape)
    max_degree_k = _extract_max_degree_and_check_shape(inputs_kv.shape)
    max_degree_qk = min(max_degree_q, max_degree_k)
    has_pseudotensors_q = inputs_q.shape[-3] == 2
    has_pseudotensors_k = inputs_kv.shape[-3] == 2
    has_pseudotensors_qk = has_pseudotensors_q and has_pseudotensors_k
    query_inputs = change_max_degree_or_type(
        inputs_q,
        max_degree=max_degree_qk,
        include_pseudotensors=has_pseudotensors_qk,
    )
    key_inputs = change_max_degree_or_type(
        inputs_kv,
        max_degree=max_degree_qk,
        include_pseudotensors=has_pseudotensors_qk,
    )

    # Query, key and value projections.
    query = Dense(
        features=qkv_features,
        use_bias=self.query_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.query_kernel_init,
        bias_init=self.query_bias_init,
        name='query',
    )(query_inputs)
    key = Dense(
        features=qkv_features,
        use_bias=self.key_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.key_kernel_init,
        bias_init=self.key_bias_init,
        name='key',
    )(key_inputs)
    value = Dense(
        features=qkv_features,
        use_bias=self.value_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.value_kernel_init,
        bias_init=self.value_bias_init,
        name='value',
    )(inputs_kv)

    # Split heads -> shape=(..., parity, degrees, features, heads).
    query = jnp.reshape(query, (*query.shape[:-1], -1, self.num_heads))
    key = jnp.reshape(key, (*key.shape[:-1], -1, self.num_heads))

    # Scale query by 1/sqrt(depth) to normalize the dot product.
    depth = math.prod(query.shape[-4:-1])  # parity * degrees * features
    query /= jnp.sqrt(depth).astype(query.dtype)

    # Gather queries and keys according to index lists.
    query = ops.gather_dst(query, adj_idx=adj_idx, dst_idx=dst_idx)
    key = ops.gather_src(key, adj_idx=adj_idx, src_idx=src_idx)

    # Dot product.
    if self.use_relative_positional_encoding_qk:
      # Compute the relative positional encoding for queries and keys from the
      # p=0, l=0 component of the basis.
      num_parity_channels = 2 if has_pseudotensors_qk else 1
      rel_pos_encoding = nn.Dense(
          features=num_parity_channels * (max_degree_qk + 1) * qkv_features,
          use_bias=self.use_basis_bias,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          kernel_init=self.dense_kernel_init,
          bias_init=self.dense_bias_init,
          name='relative_positional_encoding',
      )(basis[..., 0, 0, :])

      # Reshape to (..., num_parity_channels, max_degree+1, qkv_features).
      rel_pos_encoding = jnp.reshape(
          rel_pos_encoding,
          (
              *rel_pos_encoding.shape[:-1],
              num_parity_channels,
              max_degree_qk + 1,
              qkv_features,
          ),
      )

      # Duplicate entries for individual degrees to get the shape:
      # (..., num_parity_channels, (max_degree+1)**2, qkv_features).
      with jax.ensure_compile_time_eval():
        idx = _duplication_indices_for_max_degree(max_degree_qk)
      rel_pos_encoding = jnp.take(
          rel_pos_encoding, idx, axis=-2, indices_are_sorted=True
      )

      # Split heads -> shape=(..., parity, degrees, features, heads).
      rel_pos_encoding = jnp.reshape(
          rel_pos_encoding, (*rel_pos_encoding.shape[:-1], -1, self.num_heads)
      )

      # Position encoding weighted dot product.
      dot = jnp.einsum(
          '...plfh,...plfh,...plfh->...h',
          query,
          key,
          rel_pos_encoding,
          precision=self.precision,
          optimize='optimal',
      )
      max_attention_val = jnp.array(3.5, dtype = dot.dtype)
      dot = max_attention_val * jnp.tanh(dot / max_attention_val)
    else:
      # Normal dot product.
      dot = jnp.einsum(
          '...plfh,...plfh->...h',
          query,
          key,
          precision=self.precision,
          optimize='optimal',
      )
      max_attention_val = jnp.array(3.5, dtype = dot.dtype)
      dot = max_attention_val * jnp.tanh(dot / max_attention_val)

    # Auto-determine num segments and where mask for indexed ops (if not given).
    if num_segments is None:
      num_segments = inputs_q.shape[-4]
    if where is None and adj_idx is not None:
      where = adj_idx < inputs_kv.shape[-4]

    # Attention weights.
    weight = jax.vmap(
        functools.partial(
            ops.indexed_softmax,
            multiplicative_mask=cutoff_value,
            adj_idx=adj_idx,
            where=where,
            dst_idx=dst_idx,
            num_segments=num_segments,
            indices_are_sorted=indices_are_sorted,
        ),
        in_axes=-1,
        out_axes=-1,
    )(dot)

    # Duplicate weights for each feature in a head.
    weight = jnp.repeat(weight, qkv_features // self.num_heads, axis=-1)

    # Expand shape of weight for broadcasting (add parity and degree channel).
    weight = jnp.expand_dims(weight, (-2, -3))

    # Expand shape of value by gathering, so that it matches shape of weight.
    value = ops.gather_src(inputs=value, adj_idx=adj_idx, src_idx=src_idx)

    # Attention weighted values (with optional relative positional encoding).
    attention = super().__call__(
        inputs=weight * value,
        basis=basis if self.use_relative_positional_encoding_v else None,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )

    # Linear combination of individual attention heads.
    outputs = Dense(
        features=out_features,
        use_bias=self.output_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.output_kernel_init,
        bias_init=self.output_bias_init,
        name='out',
    )(attention)

    return outputs
  
class TanhThrSelfAttention(TanhThrMultiHeadAttention):
  r"""Equivariant self-attention.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
    num_heads: Number of attention heads.
    qkv_features: Number of features used for queries, keys and values. If this
      is `None`, the same number of features as in `inputs_q` is used.
    out_features: Number of features for the output. If this is `None`, the same
      number of features as in `inputs_q` is used.
    use_relative_positional_encoding_qk: If this is `True`, relative positional
      encodings are used for computing the dot product between queries and keys.
    use_relative_positional_encoding_v: If this is `True`, a relative positional
      encoding (with respect to the queries) is used for computing the values.
    query_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing queries.
    query_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing queries.
    query_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing queries.
    key_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing keys.
    key_bias_init: Initializer function for the bias terms of the :class:`Dense`
      layer for computing keys.
    key_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing keys.
    value_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing values.
    value_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing values.
    value_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing values.
    output_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing outputs.
    output_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing outputs.
    output_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing outputs.
  """

  @nn.compact
  def __call__(
      self,
      inputs: Union[
          Float[Array, '... N 1 (max_degree+1)**2 num_features'],
          Float[Array, '... N 2 (max_degree+1)**2 num_features'],
      ],
      basis: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      cutoff_value: Optional[
          Union[
              Float[Array, '... N M 1 #(basis_max_degree+1)**2 #num_basis'],
              Float[Array, '... P 1 #(basis_max_degree+1)**2 #num_basis'],
          ]
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      src_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (max_degree+1)**2 num_features'],
      Float[Array, '... N 2 (max_degree+1)**2 num_features'],
  ]:
    """Applies self-attention.

    In principle, self-attention is very similar to message-passing, but
    with an
    additional weight factor for each summand, with the weights summing up
    to 1.
    In contrast, in ordinary message-passing, all summands have an implicit
    weight of 1.

    Args:
      inputs: A set of :math:`N` input features.
      basis: Basis functions for all relevant interactions between pairs
        :math:`i` and :math:`j` from the :math:`N` inputs (either in dense or
        sparse indexed format).
      cutoff_value: Multiplicative cutoff values that are applied to the "raw"
        softmax values (before normalization), can be used for smooth cutoffs.
      adj_idx: Adjacency indices (dense index list), or `None`.
      where:  Mask to specify which values to sum over (only for dense index
        lists). If this is `None`, the `where` mask is auto-determined from
        `inputs`.
      dst_idx:  Destination indices (sparse index list), or `None`.
      src_idx: Source indices (sparse index list), or `None`.
      num_segments: Number of segments after summation (only for sparse index
        lists). If this is `None`, `num_segments` is auto-determined from
        `inputs`.
      indices_are_sorted: If `True`, `dst_idx` is assumed to be sorted, which
        may increase performance (only used for sparse index lists).

    Returns:
      The output of self-attention.
    """
    return super().__call__(
        inputs_q=inputs,
        inputs_kv=inputs,
        basis=basis,
        cutoff_value=cutoff_value,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        src_idx=src_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )
  