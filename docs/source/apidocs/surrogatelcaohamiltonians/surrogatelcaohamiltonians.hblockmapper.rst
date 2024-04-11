:py:mod:`surrogatelcaohamiltonians.hblockmapper`
================================================

.. py:module:: surrogatelcaohamiltonians.hblockmapper

.. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BlockIrrepMappingSpec <surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec
          :summary:
   * - :py:obj:`MultiElementPairHBlockMapper <surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`make_mapper_from_elements <surrogatelcaohamiltonians.hblockmapper.make_mapper_from_elements>`
     - .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.make_mapper_from_elements
          :summary:

API
~~~

.. py:class:: BlockIrrepMappingSpec
   :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec

   .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec

   .. py:attribute:: block_slices
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.block_slices
      :type: list[tuple[slice, slice]]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.block_slices

   .. py:attribute:: cgc_slices
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.cgc_slices
      :type: list[tuple[slice, slice, slice]]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.cgc_slices

   .. py:attribute:: irreps_slices
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.irreps_slices
      :type: list[tuple[int, slice, int]]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.irreps_slices

   .. py:attribute:: max_ell
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.max_ell
      :type: int
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.max_ell

   .. py:attribute:: nfeatures
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.nfeatures
      :type: int
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.nfeatures

   .. py:attribute:: cgc
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.cgc
      :value: 'clebsch_gordan(...)'

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.cgc

   .. py:method:: __repr__()
      :canonical: surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec.__repr__

.. py:class:: MultiElementPairHBlockMapper
   :canonical: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper

   .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper

   .. py:attribute:: mapper
      :canonical: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.mapper
      :type: dict[tuple[int, int], surrogatelcaohamiltonians.hblockmapper.BlockIrrepMappingSpec]
      :value: None

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.mapper

   .. py:method:: hblock_to_irrep(hblock, irreps_array, Z_i, Z_j)
      :canonical: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.hblock_to_irrep

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.hblock_to_irrep

   .. py:method:: hblocks_to_irrep(hblocks, irreps_array, Z_i, Z_j)
      :canonical: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.hblocks_to_irrep

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.hblocks_to_irrep

   .. py:method:: irrep_to_hblock(hblock, irreps_array, Z_i, Z_j)
      :canonical: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.irrep_to_hblock

      .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.MultiElementPairHBlockMapper.irrep_to_hblock

.. py:function:: make_mapper_from_elements(species_ells_dict: dict[int, list[int]])
   :canonical: surrogatelcaohamiltonians.hblockmapper.make_mapper_from_elements

   .. autodoc2-docstring:: surrogatelcaohamiltonians.hblockmapper.make_mapper_from_elements
