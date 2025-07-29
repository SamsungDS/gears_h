from dataclasses import field
from typing import Callable, Union

import e3x
import flax.linen as nn
from jaxtyping import Array, Float

# from slh.layers.activation import stl


class DenseBlock(nn.Module):
    dense_layer: e3x.nn.Dense = e3x.nn.Dense
    layer_widths: list[int] = field(default_factory=lambda: [128, 128])
    activation: Union[Callable, nn.Module] = e3x.nn.shifted_softplus
    name = "residual_dense"

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
        y0 = self.dense_layer(features=self.layer_widths[0], name="dense_0")(x)

        for i, width in enumerate(self.layer_widths[1:]):
            y = self.activation(y0 if i == 0 else y)  # noqa: F821
            y = self.dense_layer(features=width, name=f"dense_{i+1}")(y)

        return self.param("resid_weight", nn.initializers.constant(0.1), shape=(1,)) * y0 + y if len(self.layer_widths) > 1 else y0
