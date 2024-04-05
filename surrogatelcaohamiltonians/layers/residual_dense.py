from functools import partial

import flax.linen as nn
from typing import Union

import e3x


class DenseBlock(nn.Module):
    num_layers: int = 2
    layer_width: int = 128

    @nn.compact
    def __call__(self, x):
        y = e3x.nn.Dense(features=self.layer_width)(x)
        for _ in range(self.num_layers - 1):
            y = e3x.nn.Dense(features=self.layer_width)(y)
        return y
