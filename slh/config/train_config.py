from pathlib import Path
from typing import List, Literal, Optional, Union
from typing_extensions import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    create_model,
    model_validator,
)


class DataConfig(BaseModel, extra="forbid"):
    directory: str = "slhmodels"
    experiment: str = "default"
    # ds_type: Literal["cached", "otf"] = "cached"
    data_path: Optional[str] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    n_train: PositiveInt = 1000
    n_valid: PositiveInt = 100
    batch_size: PositiveInt = 32
    valid_batch_size: PositiveInt = 100
    shuffle_buffer_size: PositiveInt = 1000
    additional_properties_info: dict[str, str] = {}

    # shift_method: str = "per_element_regression_shift"
    # shift_options: dict = {"energy_regularisation": 1.0}

    # scale_method: str = "per_element_force_rms_scale"
    # scale_options: Optional[dict] = {}

    pos_unit: Optional[str] = "Ang"
    energy_unit: Optional[str] = "eV"

    @model_validator(mode="after")
    def set_data_or_train_val_path(self):
        not_data_path = self.data_path is None
        not_train_path = self.train_data_path is None

        neither_set = not_data_path and not_train_path
        both_set = not not_data_path and not not_train_path

        if neither_set or both_set:
            raise ValueError("Please specify either data_path or train_data_path")

        return self

    # @model_validator(mode="after")
    # def validate_shift_scale_methods(self):
    #     method_lists = [shift_method_list, scale_method_list]
    #     requested_methods = [self.shift_method, self.scale_method]
    #     requested_options = [self.shift_options, self.scale_options]

    #     cases = zip(method_lists, requested_methods, requested_options)
    #     for method_list, requested_method, requested_params in cases:
    #         methods = {method.name: method for method in method_list}

    #         # check if method exists
    #         if requested_method not in methods.keys():
    #             raise KeyError(
    #                 f"The initialization method '{requested_method}' is not among the"
    #                 f" implemented methods. Choose from {methods.keys()}"
    #             )

    #         # check if parameters names are complete and correct
    #         method = methods[requested_method]
    #         fields = {
    #             name: (dtype, ...)
    #             for name, dtype in zip(method.parameters, method.dtypes)
    #         }
    #         MethodConfig = create_model(
    #             f"{method.name}Config", __config__=ConfigDict(extra="forbid"), **fields
    #         )

    #         _ = MethodConfig(**requested_params)

    #     return self

    @property
    def model_version_path(self):
        version_path = Path(self.directory) / self.experiment
        return version_path

    @property
    def best_model_path(self):
        return self.model_version_path / "best"


class RadialBasisConfig(BaseModel, extra="forbid"):
    cutoff: PositiveFloat
    num_radial: PositiveInt
    max_degree: NonNegativeInt
    num_elemental_embedding: PositiveInt
    tensor_module: Literal["fused_tensor", "tensor"] = "tensor"

class SAAtomCenteredDescriptorConfig(BaseModel, extra="forbid"):
    descriptor_name = Literal["SAAtomCenteredDescriptor"]
    use_fused_tensor: Optional[bool] = False
    embedding_residual_connection: Optional[bool] = True
    mp_steps: Optional[int] = 2
    mp_degree: Optional[int] = 4
    mp_options: Optional[dict] = {}

class TDSAAtomCenteredDescriptorConfig(BaseModel, extra="forbid"):
    descriptor_name = Literal["TDSAAtomCenteredDescriptor"]
    max_tensordense_degree: Optional[int] = 4
    num_tensordense_features: Optional[int] = 32
    use_fused_tensor: Optional[bool] = False
    embedding_residual_connection: Optional[bool] = False

class AtomCenteredConfig(BaseModel, extra="forbid"):
    descriptor: Union[SAAtomCenteredDescriptorConfig, 
                      TDSAAtomCenteredDescriptorConfig] = Field(..., discriminator='descriptor_name')
    radial_basis: RadialBasisConfig

class BondCenteredConfig(BaseModel, extra="forbid"):
    cutoff: PositiveFloat
    max_basis_degree: NonNegativeInt = 2
    max_degree: NonNegativeInt = 4
    max_actp_degree : NonNegativeInt = 4
    tensor_module: Literal["fused_tensor", "tensor"] = "tensor"


class MLPConfig(BaseModel, extra="forbid"):
    nn: List[PositiveInt] = [128]


class ModelConfig(BaseModel, extra="forbid"):
    atom_centered: AtomCenteredConfig
    bond_centered: BondCenteredConfig
    mlp: MLPConfig

    def get_dict(self):
        return self.model_dump()


class CSVCallback(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the CSVCallback.

    Parameters
    ----------
    name: Keyword of the callback used..
    """

    name: Literal["csv"]

CallBack = Annotated[CSVCallback, Field(discriminator="name")]

# CallBack = Annotated[ # TODO implement other callbacks if we want them
#     Union[CSVCallback, TBCallback, MLFlowCallback], Field(discriminator="name")
# ]

class OptimizerConfig(BaseModel, frozen=True, extra="forbid"):
    name: str = "adam"
    lr: NonNegativeFloat = 0.001
    opt_kwargs: dict = {}
    sam_rho: NonNegativeFloat = 0.0


class TrainConfig(BaseModel, frozen=True, extra="forbid"):
    n_epochs: PositiveInt
    patience: Optional[PositiveInt] = None
    seed: int = 2465

    model: ModelConfig
    data: DataConfig
    # metrics: List[MetricsConfig] = []
    # loss: List[LossConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    callbacks: List[CallBack] = [CSVCallback(name="csv")]

    def dump_config(self, save_path: Path):
        """
        Writes the current config file to the specified directory.

        Parameters
        ----------
        save_path: Path to the directory.
        """
        with open(save_path / "config.yaml", "w") as conf:
            yaml.dump(self.model_dump(), conf, default_flow_style=False)
