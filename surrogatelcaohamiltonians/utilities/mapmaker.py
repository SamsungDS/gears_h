import numpy as np
import e3x


def get_mapping_spec(ells1, ells2):
    """I hate this function. It takes a general hamiltonian block with an
    arbitrary arragement of angular momentum blocks, and then spits out indices
    corresponding to those blocks, and the corresponding Clebsch-Gordan blocks
    (in e3x convention) that will be required to move between those and irreps.
    It also provides a mask, and slices of that mask corresponding to the
    irreps of each subblock of H. As far as I can tell, this is strictly all
    the information required to go back and forth between H blocks and irreps.

    Parameters
    ----------
    ells1 : _type_
        _description_
    ells2 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    parity_dict = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}
    # This is the maximum irrep angular momentum from Wigner-Eckhart
    # https://e3x.readthedocs.io/stable/overview.html#coupling-irreps
    max_ell = max(ells1) + max(ells2)
    # The feature direction adds up the two angular momentum numbers. This is a
    # very pessimistic estimate of the number of features required, but is
    # therefore always valid.
    mask_candidate = np.zeros(
        (2, (max_ell + 1) ** 2, len(ells1) * len(ells2)), dtype=int
    )

    ifeaturemax = 0
    colstart = 0
    rowstart = 0

    # List of tuples, each of which is the indexing for the H blocks
    block_slices = []
    # As above but for CGC blocks
    cgc_slices = []
    # As above but for final equivariant feature block
    irreps_array_slices = []

    for ell1 in ells1:
        for ell2 in ells2:
            # Is this specific product odd or even? We are rigorous about the
            # parity block mapping
            parity = int(np.logical_xor(parity_dict[ell1], parity_dict[ell2]))

            # Wigner-Eckhart irreps limits
            ellmin = np.abs(ell1 - ell2)
            ellmax = ell1 + ell2

            # Indexing for CGC subblock
            cgc_slices.append(
                (
                    slice(ell1**2, (ell1 + 1) ** 2),
                    slice(ell2**2, (ell2 + 1) ** 2),
                    slice(ellmin**2, (ellmax + 1) ** 2),
                )
            )

            # Indexing for this H subblock
            block_slices.append(
                (
                    slice(rowstart, rowstart + 2 * ell1 + 1),
                    slice(colstart, colstart + 2 * ell2 + 1),
                )
            )

            # Find first feature which has all required irreps for this feature block
            # available
            for ifeature in range(mask_candidate.shape[-1]):
                if np.all(
                    mask_candidate[parity, ellmin**2 : (ellmax + 1) ** 2, ifeature]
                    == 0
                ):
                    mask_candidate[
                        parity, ellmin**2 : (ellmax + 1) ** 2, ifeature
                    ] = 1

                    # Keep track of the maximum feature size we are at
                    # We prune the mask in the feature dimension to this
                    ifeaturemax = max(ifeaturemax, ifeature)

                    break

            # Indexing for this H block's irreps
            irreps_array_slices.append(
                (parity, slice(ellmin**2, (ellmax + 1) ** 2), ifeature)
            )

            colstart += 2 * ell2 + 1
        rowstart += 2 * ell1 + 1
        colstart = 0

    assert len(block_slices) == len(irreps_array_slices) == len(cgc_slices)
    return (block_slices, irreps_array_slices, cgc_slices, max_ell, ifeaturemax + 1)
