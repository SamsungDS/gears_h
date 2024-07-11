from typing import Literal
from pydantic import BaseModel, NonNegativeFloat

class LinearLR(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    opt_name : str, default = "adam"
    transition_begin: int = 0
        Number of steps after which to start decreasing
    end_value: NonNegativeFloat = 1e-6
        Final LR at the end of training.
    """

    name: Literal["linear"]
    transition_begin: int = 0
    end_value: NonNegativeFloat = 1e-4


class CyclicCosineLR(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    period: int = 20
        Length of a cycle.
    decay_factor: NonNegativeFloat = 1.0
        Factor by which to decrease the LR after each cycle.
        1.0 means no decrease.
    """

    name: Literal["cyclic_cosine"]
    period: int = 20
    decay_factor: NonNegativeFloat = 1.0
