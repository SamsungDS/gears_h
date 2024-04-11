:py:mod:`surrogatelcaohamiltonians.layers.residual_dense`
=========================================================

.. py:module:: surrogatelcaohamiltonians.layers.residual_dense

.. autodoc2-docstring:: surrogatelcaohamiltonians.layers.residual_dense
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DenseBlock <surrogatelcaohamiltonians.layers.residual_dense.DenseBlock>`
     -

API
~~~

.. py:class:: DenseBlock
   :canonical: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: dense_layer
      :canonical: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.dense_layer
      :type: flax.linen.Module
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.dense_layer

   .. py:attribute:: layer_widths
      :canonical: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.layer_widths
      :type: list[int]
      :value: 'field(...)'

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.layer_widths

   .. py:method:: __call__(x)
      :canonical: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.__call__

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.residual_dense.DenseBlock.__call__
