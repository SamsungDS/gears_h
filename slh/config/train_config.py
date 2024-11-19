from pathlib import Path
from typing import List, Literal, Optional, Union, Any
from typing_extensions import Annotated

import yaml
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator
)

from slh.config.lr_config import (LinearSchedule, 
                                  CyclicCosineSchedule, 
                                  ExponentialDecaySchedule,
                                  WarmupCosineDecay
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

    bond_fraction: PositiveFloat = 1.0

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

    @property
    def model_version_path(self):
        version_path = Path(self.directory) / self.experiment
        return version_path

    @property
    def best_model_path(self):
        return self.model_version_path / "best"


class RadialBasisConfig(BaseModel, extra="forbid"):
    cutoff: PositiveFloat
    num_radial: PositiveInt = 8
    max_degree: NonNegativeInt = 2
    num_elemental_embedding: PositiveInt = 8
    embedding_residual_connection: bool = False
    tensor_module: Literal["fused_tensor", "tensor"] = "tensor"
    tensor_module_dtype: Literal["float32", "float64", "bfloat16"] = "float32"

class SAAtomCenteredDescriptorConfig(BaseModel, extra="forbid"):
    descriptor_name: Literal["SAAtomCenteredDescriptor"] = "SAAtomCenteredDescriptor"
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = True
    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = {}

class TDSAAtomCenteredDescriptorConfig(BaseModel, extra="forbid"):
    descriptor_name: Literal["TDSAAtomCenteredDescriptor"] = "TDSAAtomCenteredDescriptor"
    max_tensordense_degree: int = 4
    num_tensordense_features: int = 32
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = False
    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = {}

class ShallowTDSAAtomCenteredDescriptorConfig(BaseModel, extra="forbid"):
    descriptor_name: Literal["ShallowTDSAAtomCenteredDescriptor"] = "ShallowTDSAAtomCenteredDescriptor"
    max_tensordense_degree: int = 4
    num_tensordense_features: int = 32
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = False
    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = {}
    mp_basis_options: dict[str, str | int | dict] = {"radial_fn" : "basic_fourier",
                                                     "radial_kwargs" : {},
                                                     "max_degree" : 2,
                                                     "num" : 8,
                                                     "cutoff_fn" : "smooth_cutoff"
                                                    }

class SlightlyDifferentShallowTDSAAtomCenteredDescriptor(BaseModel, extra="forbid"):
    descriptor_name: Literal["SlightlyDifferentShallowTDSAAtomCenteredDescriptor"] = "SlightlyDifferentShallowTDSAAtomCenteredDescriptor"
    max_tensordense_degree: int = 4
    num_tensordense_features: int = 32
    use_fused_tensor: bool = False
    embedding_residual_connection: bool = False
    mp_steps: int = 2
    mp_degree: int = 4
    mp_options: dict = {}
    mp_basis_options: dict[str, str | int | dict] = {"radial_fn" : "basic_fourier",
                                                     "radial_kwargs" : {},
                                                     "max_degree" : 2,
                                                     "num" : 8,
                                                     "cutoff_fn" : "smooth_cutoff"
                                                    }

class AtomCenteredConfig(BaseModel, extra="forbid"):
    descriptor: Union[SAAtomCenteredDescriptorConfig, 
                      TDSAAtomCenteredDescriptorConfig,
                      ShallowTDSAAtomCenteredDescriptorConfig,
                      SlightlyDifferentShallowTDSAAtomCenteredDescriptor
                     ] = Field(SAAtomCenteredDescriptorConfig(),
                               discriminator='descriptor_name')
    radial_basis: RadialBasisConfig

class BondCenteredConfig(BaseModel, extra="forbid"):
    bond_expansion_options: dict[str, str | int | dict] = {"radial_fn" : "basic_fourier",
                                                           "radial_kwargs" : {},
                                                           "cutoff_fn" : "smooth_cutoff",
                                                           "max_degree" : 2,
                                                           "num" : 8
                                                          }
    cutoff: PositiveFloat
    max_basis_degree: NonNegativeInt = 2
    max_degree: NonNegativeInt = 4
    tensor_module: Literal["fused_tensor", "tensor"] = "tensor"
    tensor_module_dtype: Literal["float32", "float64", "bfloat16"] = "float32"


class MLPConfig(BaseModel, extra="forbid"):
    mlp_layer_widths: List[PositiveInt] = [128]
    mlp_dtype: Literal["float32", "float64", "bfloat16"] = "float32"
    mlp_activation_function: Literal["shifted_softplus", 
                                     "mish",
                                     "bent_identity"] = "shifted_softplus"


class ModelConfig(BaseModel, extra="forbid"):
    atom_centered: AtomCenteredConfig
    bond_centered: BondCenteredConfig
    mlp: MLPConfig

class CSVCallback(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the CSVCallback.

    Parameters
    ----------
    name: Keyword of the callback used..
    """

    name: Literal["csv"]

CallBack = Annotated[CSVCallback, Field(discriminator="name")]

# CallBack = Annotated[ # TODO implement other   callbacks if we want them
#     Union[CSVCallback, TBCallback, MLFlowCallback], Field(discriminator="name")
# ]

class OptimizerConfig(BaseModel, frozen=True, extra="forbid"):
    name: str = "adam"
    lr: NonNegativeFloat = 0.005
    opt_kwargs: dict[str, Any] = {"nesterov" : True}
    schedule: Union[LinearSchedule, 
                    CyclicCosineSchedule, 
                    ExponentialDecaySchedule,
                    WarmupCosineDecay] = Field(ExponentialDecaySchedule(), 
                                               discriminator="name")
    
class LossConfig(BaseModel, frozen=True, extra="forbid"):
    name: str = "weighted_mse_and_rmse"
    loss_parameters: dict[str, NonNegativeFloat] = {"off_diagonal_weight" : 4.0,
                                         "on_diagonal_weight" : 1.0,
                                         "mse_wweight" : 1.0,
                                         "rmse_weight" : 1.0,
                                         "loss_multiplier" : 5.0,
                                         "alpha" : Field(default=0.9, le=1, ge=0)
                                        }

class TrainConfig(BaseModel, frozen=True, extra="forbid"):
    n_epochs: PositiveInt
    patience: Optional[PositiveInt] = None
    seed: int = 2465

    model: ModelConfig
    data: DataConfig
    # metrics: List[MetricsConfig] = []
    loss: LossConfig
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
