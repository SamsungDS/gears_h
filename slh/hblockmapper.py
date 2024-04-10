from dataclasses import dataclass
from functools import cache
from itertools import product
from slh.utilities.mapmaker import get_mapping_spec

import numpy as np

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

    def hblocks_to_irrep(self, hblocks, irreps_array, Z_i, Z_j):
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

    def irrep_to_hblock(self, hblock, irreps_array, Z_i, Z_j):
        mapping_spec = self.mapper[(Z_i, Z_j)]

        ms = mapping_spec
        for block_slice, cgc_slice, irreps_slice in zip(
            ms.block_slices, ms.cgc_slices, ms.irreps_slices, strict=True
        ):
            hblock[block_slice] = np.einsum(
                "l,mnl->mn", irreps_array[irreps_slice], ms.cgc[cgc_slice]
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
