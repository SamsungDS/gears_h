:py:mod:`slh.layers.descriptor.radial_basis`
============================================

.. py:module:: slh.layers.descriptor.radial_basis

.. autodoc2-docstring:: slh.layers.descriptor.radial_basis
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SpeciesAwareRadialBasis <slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`jinclike <slh.layers.descriptor.radial_basis.jinclike>`
     - .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.jinclike
          :summary:

API
~~~

.. py:class:: SpeciesAwareRadialBasis
   :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: cutoff
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.cutoff
      :type: float
      :value: None

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.cutoff

   .. py:attribute:: num_radial
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_radial
      :type: int
      :value: 8

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_radial

   .. py:attribute:: max_degree
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.max_degree
      :type: int
      :value: 3

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.max_degree

   .. py:attribute:: num_elemental_embedding
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_elemental_embedding
      :type: int
      :value: 64

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_elemental_embedding

   .. py:attribute:: tensor_module
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.tensor_module
      :type: typing.Union[e3x.nn.Tensor, e3x.nn.FusedTensor]
      :value: None

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.tensor_module

   .. py:attribute:: embedding_residual_connection
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.embedding_residual_connection
      :type: int
      :value: True

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.embedding_residual_connection

   .. py:method:: setup()
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.setup

   .. py:method:: __call__(neighbour_displacements: jaxtyping.Float[jaxtyping.Array, ... num_neighbours 3], Z_j: jaxtyping.Float[jaxtyping.Array, ... num_neighbours])
      :canonical: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.__call__

      .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.__call__

.. py:function:: jinclike(x: jaxtyping.Float[jaxtyping.Array, ...], num: int, limit: float = 1.0)
   :canonical: slh.layers.descriptor.radial_basis.jinclike

   .. autodoc2-docstring:: slh.layers.descriptor.radial_basis.jinclike
