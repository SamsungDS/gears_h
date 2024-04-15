from dataclasses import dataclass
from slh.utilities.mapmaker import get_mapping_spec

import numpy as np

from itertools import product
import logging

log = logging.getLogger(__name__)

import e3x


@dataclass(frozen=True)
class BlockIrrepMappingSpec:
    # Slices for subblocking a block
    block_slices: list[tuple[slice, slice]]
    # Slices of CG coeffs to transform back and forth
    cgc_slices: list[tuple[slice, slice, slice]]
    # Slices of irreps array corresponding to subblocks
    irreps_slices: list[tuple[int, slice, int]]
    max_ell: int
    nfeatures: int
    cgc = e3x.so3.clebsch_gordan(2, 2, 4)

    def __repr__(self):
        return f"Mapper(nblocks={len(self.block_slices)}, max_ell={self.max_ell}, nfeatures={self.nfeatures})"


@dataclass(frozen=True)
class MultiElementPairHBlockMapper:
    # We keep atomic_number_pairs to map onto the hamiltonian block mappers
    mapper: dict[tuple[int, int], BlockIrrepMappingSpec]

    def hblock_to_irrep(self, hblock, irreps_array, Z_i, Z_j):
        mapping_spec = self.mapper[(Z_i, Z_j)]

        ms = mapping_spec
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            np.einsum(
                "mn,mnl->l",
                hblock[block_slice],
                ms.cgc[cgc_slice],
                out=irreps_array[irreps_slice],
            )

    def hblocks_to_irreps(self, hblocks, irreps_array, Z_i, Z_j):
        assert len(hblocks) == len(irreps_array)
        mapping_spec = self.mapper[(Z_i, Z_j)]

        ms = mapping_spec
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            block_slice = (slice(0, len(hblocks)),) + block_slice
            irreps_slice = (slice(0, len(hblocks)),) + irreps_slice

            np.einsum(
                "...mn,mnl->...l",
                hblocks[block_slice],
                ms.cgc[cgc_slice],
                out=irreps_array[irreps_slice],
                optimize=True,
            )
        return irreps_array

    def irreps_to_hblocks(self, hblocks, irreps_array, Z_i, Z_j):
        mapping_spec = self.mapper[(Z_i, Z_j)]

        ms = mapping_spec
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            block_slice = (slice(0, len(hblocks)),) + block_slice
            irreps_slice = (slice(0, len(hblocks)),) + irreps_slice
            hblocks[block_slice] = np.einsum(
                "...l,mnl->...mn", irreps_array[irreps_slice], ms.cgc[cgc_slice]
            )


def make_mapper_from_elements(species_ells_dict: dict[int, list[int]]):
    element_pair_list = []
    hblock_mapper_list = []

    atomic_numbers = species_ells_dict.keys()
    for Z_i, Z_j in product(atomic_numbers, atomic_numbers):
        ells1 = species_ells_dict[Z_i]
        ells2 = species_ells_dict[Z_j]
        (
            block_slices,
            irreps_slices,
            cgc_slices,
            max_ell_for_pair,
            num_features_for_pair,
        ) = get_mapping_spec(ells1, ells2)
        
        log.info(f"Pair: {Z_i}, {Z_j}, max_ell: {max_ell_for_pair}, num_features:{num_features_for_pair}")
        
        element_pair_list.append((Z_i, Z_j))
        hblock_mapper_list.append(
            BlockIrrepMappingSpec(
                block_slices=block_slices,
                cgc_slices=cgc_slices,
                irreps_slices=irreps_slices,
                max_ell=max_ell_for_pair,
                nfeatures=num_features_for_pair,
            )
        )
    return MultiElementPairHBlockMapper(
        dict(zip(element_pair_list, hblock_mapper_list))
    )


def get_mask_dict(
    max_ell: int, nfeatures: int, pairwise_hmap: MultiElementPairHBlockMapper
) -> dict[tuple[int, int], np.ndarray]:
    mask_dict = {}
    for element_pair, blockmapper in pairwise_hmap.mapper.items():
        # This is e3x convention. 2 for parity, angular momentum channels, features
        mask_array = np.zeros((2, (max_ell + 1) ** 2, nfeatures), dtype=np.int8)
        for _slice in blockmapper.irreps_slices:
            log.debug(f"Element pair: {element_pair}, mask:\n{_slice}")
            mask_array[_slice] = 1
        mask_dict[element_pair] = mask_array

        log.debug(f"Element pair: {element_pair}, mask:\n{mask_array}")
    return mask_dict
