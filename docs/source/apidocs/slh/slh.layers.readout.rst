:py:mod:`slh.layers.readout`
============================

.. py:module:: slh.layers.readout

.. autodoc2-docstring:: slh.layers.readout
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Readout <slh.layers.readout.Readout>`
     -

API
~~~

.. py:class:: Readout
   :canonical: slh.layers.readout.Readout

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: features
      :canonical: slh.layers.readout.Readout.features
      :type: int
      :value: None

      .. autodoc2-docstring:: slh.layers.readout.Readout.features

   .. py:attribute:: max_ell
      :canonical: slh.layers.readout.Readout.max_ell
      :type: int
      :value: 4

      .. autodoc2-docstring:: slh.layers.readout.Readout.max_ell

   .. py:method:: __call__(y)
      :canonical: slh.layers.readout.Readout.__call__

      .. autodoc2-docstring:: slh.layers.readout.Readout.__call__
