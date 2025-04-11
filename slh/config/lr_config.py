from typing import Literal
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt


class ConstantSchedule(BaseModel, frozen=True, extra="forbid"):
    """
    Constant LR scheduler.

    Parameters
    ----------
    """

    name: Literal["constant_schedule"] = "constant_schedule"

class LinearSchedule(BaseModel, frozen=True, extra="forbid"):
    """
    Configuration of the optimizer.
    Learning rates of 0 will freeze the respective parameters.

    Parameters
    ----------
    transition_begin: int = 0
        Number of steps after which to start decreasing
    transition_steps: int
        Number of steps it takes to go from the initial LR to the final one, starting from transition_begin.
    end_value: NonNegativeFloat = 1e-6
        Final LR at the end of training.
    """

    name: Literal["linear_schedule"] = "linear_schedule"
    transition_begin: int = 0
    transition_steps: int = 1000
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

class ReduceOnPlateauSchedule(BaseModel, frozen = True, extra = "forbid"):
    name: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    factor: NonNegativeFloat = 0.5
    patience: NonNegativeInt = 10
    rtol: NonNegativeFloat = 0.0001
    atol: float = 0.0
    accumulation_size: NonNegativeInt = 1
    min_scale: NonNegativeFloat = 1e-5
    cooldown: NonNegativeInt = 0

class WarmupExponentialDecaySchedule(BaseModel, frozen = True, extra = "forbid"):
    name: Literal["warmup_exponential_decay_schedule"] = "warmup_exponential_decay_schedule"
    peak_value: NonNegativeFloat =5e-3
    warmup_steps: NonNegativeInt = 5000
    transition_begin: NonNegativeInt = 2000
    transition_steps: NonNegativeInt = 1
    end_value: NonNegativeFloat = 5e-5
    decay_rate: NonNegativeInt = 0.99995
    staircase: bool = False

class WarmupCosineDecaySchedule(BaseModel, frozen = True, extra = "forbid"):
    name: Literal["warmup_cosine_decay_schedule"] = "warmup_cosine_decay_schedule"
    #init_value: NonNegativeFloat = 1e-3
    peak_value: NonNegativeFloat = 1e-2
    end_value: NonNegativeFloat = 1e-4
    warmup_steps: NonNegativeInt = 25
    decay_steps: NonNegativeInt = 225
    exponent: float = 1