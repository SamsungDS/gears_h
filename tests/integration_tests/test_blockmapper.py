from slh.hblockmapper import make_mapper_from_elements, get_mask_dict
import numpy as np


def test_block_mapper_identity():

  mapper = make_mapper_from_elements({1: [0, 1]})

  mapping_spec = mapper.mapper[(1, 1)]
  max_ell, nfeatures = mapping_spec.max_ell, mapping_spec.nfeatures
  mask_dict = get_mask_dict(max_ell, nfeatures, mapper)

  hblock = np.random.rand(1, 4, 4)
  irrepblock = np.zeros((1, 2, (max_ell + 1) ** 2, nfeatures))

  irrepblock = mapper.hblocks_to_irreps(hblock, irrepblock, 1, 1)
  irrepblock *= mask_dict[(1, 1)]

  hblock_ = np.zeros((1, 4, 4))

  mapper.irreps_to_hblocks(hblock_, irrepblock, 1, 1)
  assert np.allclose(hblock, hblock_, atol=np.finfo(np.float32).eps)


test_block_mapper_identity()