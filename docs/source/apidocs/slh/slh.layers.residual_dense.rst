:py:mod:`slh.layers.residual_dense`
===================================

.. py:module:: slh.layers.residual_dense

.. autodoc2-docstring:: slh.layers.residual_dense
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DenseBlock <slh.layers.residual_dense.DenseBlock>`
     -

API
~~~

.. py:class:: DenseBlock
   :canonical: slh.layers.residual_dense.DenseBlock

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: dense_layer
      :canonical: slh.layers.residual_dense.DenseBlock.dense_layer
      :type: flax.linen.Module
      :value: None

      .. autodoc2-docstring:: slh.layers.residual_dense.DenseBlock.dense_layer

   .. py:attribute:: layer_widths
      :canonical: slh.layers.residual_dense.DenseBlock.layer_widths
      :type: list[int]
      :value: 'field(...)'

      .. autodoc2-docstring:: slh.layers.residual_dense.DenseBlock.layer_widths

   .. py:method:: __call__(x)
      :canonical: slh.layers.residual_dense.DenseBlock.__call__

      .. autodoc2-docstring:: slh.layers.residual_dense.DenseBlock.__call__
