from slh.hblockmapper import MultiElementPairHBlockMapper, BlockIrrepMappingSpec
from slh.data.input_pipeline import get_max_ell_and_max_features

def test_max_ell_and_max_features():
  mapper = MultiElementPairHBlockMapper({(1, 1): BlockIrrepMappingSpec(None, None, None, 1, 1)})
  max_ell, max_features = get_max_ell_and_max_features(mapper)
  assert max_ell == 1
  assert max_features == 1