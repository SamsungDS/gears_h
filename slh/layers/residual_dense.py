from functools import partial
from dataclasses import field

import flax.linen as nn
from typing import Union

import e3x
from jaxtyping import Float, Array


class DenseBlock(nn.Module):
    dense_layer: e3x.nn.Dense
    layer_widths: list[int] = field(default_factory=lambda: [128, 128])

    @nn.compact
    def __call__(
        self,
        x: Union[
            Float[Array, "... 1 (max_degree+1)**2 num_features"],
            Float[Array, "... 2 (max_degree+1)**2 num_features"],
        ],
    ) -> Union[
        Float[Array, "... 1 (max_degree+1)**2 last_layer_width"],
        Float[Array, "... 2 (max_degree+1)**2 last_layer_width"],
    ]:
        y = self.dense_layer(features=self.layer_widths[0])(x)

        for width in self.layer_widths[1:]:
            y = e3x.nn.soft_sign(y)
            y = self.dense_layer(features=width)(y)

        return y
