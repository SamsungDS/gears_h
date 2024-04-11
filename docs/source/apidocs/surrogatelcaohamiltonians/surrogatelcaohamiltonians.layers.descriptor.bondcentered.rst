:py:mod:`surrogatelcaohamiltonians.layers.descriptor.bondcentered`
==================================================================

.. py:module:: surrogatelcaohamiltonians.layers.descriptor.bondcentered

.. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.bondcentered
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BondCenteredTensorMomentDescriptor <surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor>`
     -

API
~~~

.. py:class:: BondCenteredTensorMomentDescriptor
   :canonical: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor

   Bases: :py:obj:`flax.linen.Module`

   .. py:attribute:: cutoff
      :canonical: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.cutoff
      :type: float
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.cutoff

   .. py:attribute:: max_degree
      :canonical: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.max_degree
      :type: int
      :value: 4

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.max_degree

   .. py:attribute:: tensor_module
      :canonical: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.tensor_module
      :type: typing.Union[e3x.nn.Tensor, e3x.nn.FusedTensor]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.tensor_module

   .. py:method:: __call__(atomic_descriptors, neighbour_indices, neighbour_displacements)
      :canonical: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.__call__

      .. autodoc2-docstring:: surrogatelcaohamiltonians.layers.descriptor.bondcentered.BondCenteredTensorMomentDescriptor.__call__
