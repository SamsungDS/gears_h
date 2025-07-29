import logging
import os
from collections.abc import MutableMapping
from typing import Union

import yaml

from slh.config.train_config import TrainConfig

log = logging.getLogger(__name__)


def parse_config(config: Union[str, os.PathLike, dict], mode: str = "train") -> TrainConfig:
    """Load the training configuration from file or a dictionary.

    Parameters
    ----------
        config : str | os.PathLike | dict
            Path to the config file or a dictionary
            containing the config.
        mode: str, default = train
            Defines if the config is validated for training ("train")
            or MD simulation("md").
    """
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

    if mode == "train":
        config = TrainConfig.model_validate(config)

    return config
