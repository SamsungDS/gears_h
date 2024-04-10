from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    create_model,
    model_validator,
    StrictBool,
)
from typing import List, Literal


class ModelConfig(BaseModel, extra="forbid"):
    radial_cutoff: PositiveFloat
    n_radial: PositiveInt = 32
    radial_max_degree: PositiveInt = 2
    elemental_embedding_width: PositiveInt = 64
    radial_embedding_residual_connection: StrictBool = True

    num_moment_features: PositiveInt = 64
    n_max_moment: PositiveInt = 2
    moment_max_degree: PositiveInt = 4
    use_fused_tensor: StrictBool = True
    ac_embedding_residual_connection: StrictBool = True

    bc_cutoff: PositiveFloat
    bc_max_degree: PositiveInt = 2

    nn: List[PositiveInt] = [128, 128]

    ac_tensor_dtype: Literal["bf16", "fp32", "fp64"] = "fp32"
    bc_tensor_dtype: Literal["bf16", "fp32", "fp64"] = "fp32"

    def get_dict(self):
        return self.model_dump()
