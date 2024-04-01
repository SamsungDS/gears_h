from surrogatelcaohamiltonians.utilities.mapper import get_mapping_spec


def test_spsp_h_cgc_ia():
    ells1 = [0, 1]
    ells2 = [0, 1]

    block_slices, ia_slices, cgc_slices, irrep_mask = get_mapping_spec(ells1, ells2)

    cgc_mn_slices = [(x[0], x[1]) for x in cgc_slices]
    for slice_pair in [
        (slice(0, 1), slice(0, 1)),
        (slice(0, 1), slice(1, 4)),
        (slice(1, 4), slice(0, 1)),
        (slice(1, 4), slice(1, 4)),
    ]:
        assert slice_pair in block_slices
        assert slice_pair in cgc_mn_slices

    irrep_ell_slices = [x[1] for x in ia_slices]
    for _slice in [slice(0, 1), slice(1, 4), slice(0, 9)]:
        assert _slice in irrep_ell_slices


def test_spsd_h_cgc_ia():
    ells1 = [0, 1]
    ells2 = [0, 2]

    block_slices, ia_slices, cgc_slices, irrep_mask = get_mapping_spec(ells1, ells2)

    for slice_pair in [
        (slice(0, 1), slice(0, 1)),
        (slice(0, 1), slice(1, 6)),
        (slice(1, 4), slice(0, 1)),
        (slice(1, 4), slice(1, 6)),
    ]:
        assert slice_pair in block_slices

    cgc_mn_slices = [(x[0], x[1]) for x in cgc_slices]
    for slice_pair in [
        (slice(0, 1), slice(0, 1)),
        (slice(0, 1), slice(4, 9)),
        (slice(1, 4), slice(0, 1)),
        (slice(1, 4), slice(4, 9)),
    ]:
        assert slice_pair in cgc_mn_slices

    irrep_ell_slices = [x[1] for x in ia_slices]
    for irreps_slices in [slice(0, 1), slice(1, 4), slice(4, 9), slice(1, 16)]:
        assert irreps_slices in irrep_ell_slices


def test_spdsd_h_cgc_ia():
    ells1 = [0, 1, 2]
    ells2 = [0, 2]

    block_slices, ia_slices, cgc_slices, irrep_mask = get_mapping_spec(ells1, ells2)

    for slice_pair in [
        (slice(0, 1), slice(0, 1)),
        (slice(0, 1), slice(1, 6)),
        (slice(1, 4), slice(0, 1)),
        (slice(1, 4), slice(1, 6)),
        (slice(4, 9), slice(0, 1)),
        (slice(4, 9), slice(1, 6)),
    ]:
        assert slice_pair in block_slices

    cgc_mn_slices = [(x[0], x[1]) for x in cgc_slices]
    for slice_pair in [
        (slice(0, 1), slice(0, 1)),
        (slice(0, 1), slice(4, 9)),
        (slice(1, 4), slice(0, 1)),
        (slice(1, 4), slice(4, 9)),
        (slice(0, 1), slice(4, 9)),
        (slice(4, 9), slice(4, 9)),
    ]:
        assert slice_pair in cgc_mn_slices

    irrep_ell_slices = [x[1] for x in ia_slices]
    for irreps_slices in [
        slice(0, 1),
        slice(1, 4),
        slice(4, 9),
        slice(1, 16),
        slice(4, 9),
        slice(1, 16),
    ]:
        assert irreps_slices in irrep_ell_slices
