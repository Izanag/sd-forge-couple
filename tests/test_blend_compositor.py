import numpy as np

from lib_couple.blend_compositor import composite_layers
from lib_couple.region_shapes import BlendMode


def test_normal_blend_produces_union():
    top = np.full((2, 2), 0.5, dtype=np.float32)
    bottom = np.full((2, 2), 0.5, dtype=np.float32)

    result = composite_layers([(top, BlendMode.NORMAL)], base=bottom)

    assert np.allclose(result, 0.75)
