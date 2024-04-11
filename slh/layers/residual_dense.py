from functools import partial
from dataclasses import field

import flax.linen as nn
from typing import Union

import e3x
import jax.numpy as jnp


class DenseBlock(nn.Module):
    dense_layer: nn.Module
    layer_widths: list[int] = field(default_factory=lambda: [128, 128])

    @nn.compact
    def __call__(self, x):
        
        y = self.dense_layer(features=self.layer_widths[0])(x)
        
        for width in self.layer_widths[1:]:
            y = e3x.nn.swish(y)
            y = self.dense_layer(features=width)(y)
        
        return y