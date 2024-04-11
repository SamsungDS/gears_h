:py:mod:`surrogatelcaohamiltonians.layers.descriptor.radial_basis`
==================================================================

.. py:module:: surrogatelcaohamiltonians.layers.descriptor.radial_basis

.. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SpeciesAwareRadialBasis <surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`jinclike <surrogatelcaohamiltonians.layers.descriptor.radial_basis.jinclike>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.jinclike
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Array <surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array
          :summary:
   * - :py:obj:`Float <surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float
          :summary:

API
~~~

.. py:data:: Array
   :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array
   :value: None

   .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array

.. py:data:: Float
   :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float
   :value: None

   .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float

.. py:class:: SpeciesAwareRadialBasis
   :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: cutoff
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.cutoff
      :type: float
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.cutoff

   .. py:attribute:: num_radial
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_radial
      :type: int
      :value: 8

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_radial

   .. py:attribute:: max_degree
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.max_degree
      :type: int
      :value: 3

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.max_degree

   .. py:attribute:: num_elemental_embedding
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_elemental_embedding
      :type: int
      :value: 64

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.num_elemental_embedding

   .. py:attribute:: tensor_module
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.tensor_module
      :type: typing.Union[e3x.nn.Tensor, e3x.nn.FusedTensor]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.tensor_module

   .. py:attribute:: embedding_residual_connection
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.embedding_residual_connection
      :type: int
      :value: True

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.embedding_residual_connection

   .. py:method:: setup()
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.setup

   .. py:method:: __call__(neighbour_displacements: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float[surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array, ... 3], Z_j: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float[surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array, ...])
      :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.__call__

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis.__call__

.. py:function:: jinclike(x: surrogatelcaohamiltonians.layers.descriptor.radial_basis.Float[surrogatelcaohamiltonians.layers.descriptor.radial_basis.Array, ...], num: int, limit: float = 1.0)
   :canonical: surrogatelcaohamiltonians.layers.descriptor.radial_basis.jinclike

   .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.radial_basis.jinclike
