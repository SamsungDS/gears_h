from typing import Literal
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

class LinearSchedule(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    transition_begin: int = 0
        Number of steps after which to start decreasing
    end_value: NonNegativeFloat = 1e-6
        Final LR at the end of training.
    """

    name: Literal["linear_schedule"] = "linear_schedule"
    transition_begin: int = 0
    end_value: NonNegativeFloat = 1e-4


class CyclicCosineSchedule(BaseModel, frozen=True, extra="forbid"):
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

    name: Literal["cyclic_cosine"] = "cyclic_cosine"
    period: int = 20
    decay_factor: NonNegativeFloat = 1.0

class ExponentialDecaySchedule(BaseModel, frozen=True, extra="forbid"):
    name: Literal["exponential_decay"] = "exponential_decay"
    transition_steps: int = 1
    transition_begin: int = 20
    decay_rate: NonNegativeFloat = 0.99
    end_value: NonNegativeFloat = 1e-4

class WarmupCosineDecay(BaseModel, frozen = True, extra = "forbid"):
    name: Literal["warmup_cosine_decay_schedule"] = "warmup_cosine_decay_schedule"
    #init_value: NonNegativeFloat = 1e-3
    peak_value: NonNegativeFloat = 1e-2
    end_value: NonNegativeFloat = 1e-4
    warmup_steps: NonNegativeInt = 25
    decay_steps: NonNegativeInt = 225
    exponent: float = 1