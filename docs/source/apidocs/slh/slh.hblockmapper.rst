:py:mod:`slh.hblockmapper`
==========================

.. py:module:: slh.hblockmapper

.. autodoc2-docstring:: slh.hblockmapper
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BlockIrrepMappingSpec <slh.hblockmapper.BlockIrrepMappingSpec>`
     - .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec
          :summary:
   * - :py:obj:`MultiElementPairHBlockMapper <slh.hblockmapper.MultiElementPairHBlockMapper>`
     - .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`make_mapper_from_elements <slh.hblockmapper.make_mapper_from_elements>`
     - .. autodoc2-docstring:: slh.hblockmapper.make_mapper_from_elements
          :summary:
   * - :py:obj:`get_mask_dict <slh.hblockmapper.get_mask_dict>`
     - .. autodoc2-docstring:: slh.hblockmapper.get_mask_dict
          :summary:

API
~~~

.. py:class:: BlockIrrepMappingSpec
   :canonical: slh.hblockmapper.BlockIrrepMappingSpec

   .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec

   .. py:attribute:: block_slices
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.block_slices
      :type: list[tuple[slice, slice]]
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.block_slices

   .. py:attribute:: cgc_slices
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.cgc_slices
      :type: list[tuple[slice, slice, slice]]
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.cgc_slices

   .. py:attribute:: irreps_slices
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.irreps_slices
      :type: list[tuple[int, slice, int]]
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.irreps_slices

   .. py:attribute:: max_ell
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.max_ell
      :type: int
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.max_ell

   .. py:attribute:: nfeatures
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.nfeatures
      :type: int
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.nfeatures

   .. py:attribute:: cgc
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.cgc
      :value: 'clebsch_gordan(...)'

      .. autodoc2-docstring:: slh.hblockmapper.BlockIrrepMappingSpec.cgc

   .. py:method:: __repr__()
      :canonical: slh.hblockmapper.BlockIrrepMappingSpec.__repr__

.. py:class:: MultiElementPairHBlockMapper
   :canonical: slh.hblockmapper.MultiElementPairHBlockMapper

   .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper

   .. py:attribute:: mapper
      :canonical: slh.hblockmapper.MultiElementPairHBlockMapper.mapper
      :type: dict[tuple[int, int], slh.hblockmapper.BlockIrrepMappingSpec]
      :value: None

      .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper.mapper

   .. py:method:: hblock_to_irrep(hblock, irreps_array, Z_i, Z_j)
      :canonical: slh.hblockmapper.MultiElementPairHBlockMapper.hblock_to_irrep

      .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper.hblock_to_irrep

   .. py:method:: hblocks_to_irreps(hblocks, irreps_array, Z_i, Z_j)
      :canonical: slh.hblockmapper.MultiElementPairHBlockMapper.hblocks_to_irreps

      .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper.hblocks_to_irreps

   .. py:method:: irreps_to_hblocks(hblocks, irreps_array, Z_i, Z_j)
      :canonical: slh.hblockmapper.MultiElementPairHBlockMapper.irreps_to_hblocks

      .. autodoc2-docstring:: slh.hblockmapper.MultiElementPairHBlockMapper.irreps_to_hblocks

.. py:function:: make_mapper_from_elements(species_ells_dict: dict[int, list[int]])
   :canonical: slh.hblockmapper.make_mapper_from_elements

   .. autodoc2-docstring:: slh.hblockmapper.make_mapper_from_elements

.. py:function:: get_mask_dict(max_ell: int, nfeatures: int, pairwise_hmap: slh.hblockmapper.MultiElementPairHBlockMapper) -> dict[tuple[int, int], numpy.ndarray]
   :canonical: slh.hblockmapper.get_mask_dict

   .. autodoc2-docstring:: slh.hblockmapper.get_mask_dict
