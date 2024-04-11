:py:mod:`surrogatelcaohamiltonians.layers.descriptor.atomcentered`
==================================================================

.. py:module:: surrogatelcaohamiltonians.layers.descriptor.atomcentered

.. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AtomCenteredTensorMomentDescriptor <surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor>`
     -

API
~~~

.. py:class:: AtomCenteredTensorMomentDescriptor
   :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: radial_basis
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.radial_basis
      :type: surrogatelcaohamiltonians.layers.descriptor.radial_basis.SpeciesAwareRadialBasis
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.radial_basis

   .. py:attribute:: num_moment_features
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.num_moment_features
      :type: int
      :value: 64

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.num_moment_features

   .. py:attribute:: max_moment
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.max_moment
      :type: int
      :value: 2

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.max_moment

   .. py:attribute:: moment_max_degree
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.moment_max_degree
      :type: int
      :value: 4

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.moment_max_degree

   .. py:attribute:: use_fused_tensor
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.use_fused_tensor
      :type: bool
      :value: False

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.use_fused_tensor

   .. py:attribute:: embedding_residual_connection
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.embedding_residual_connection
      :type: bool
      :value: True

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.embedding_residual_connection

   .. py:method:: setup()
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.setup

   .. py:method:: __call__(atomic_numbers: jaxtyping.Int[jaxtyping.Array, num_atoms], neighbour_indices: jaxtyping.Int[jaxtyping.Array, ... num_neighbours 2], neighbour_displacements: jaxtyping.Float[jaxtyping.Array, ... num_neighbours 3])
      :canonical: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.__call__

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.atomcentered.AtomCenteredTensorMomentDescriptor.__call__
