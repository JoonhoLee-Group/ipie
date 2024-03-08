from dataclasses import dataclass
from typing import ClassVar

from ipie.qmc.options import QMCParams

_no_default = object()

@dataclass
class ThermalQMCParams(QMCParams):
    r"""Input options and certain constants / parameters derived from them.

    Args:
        beta: inverse temperature.
        num_walkers: number of walkers **per** core / task / computational unit.
        total_num_walkers: The total number of walkers in the simulation.
        timestep: The timestep delta_t
        num_steps_per_block: Number of steps of propagation before estimators
            are evaluated.
        num_blocks: The number of blocks. Total number of iterations =
            num_blocks * num_steps_per_block.
        num_stblz: number of steps before QR stabilization of walkers is performed.
        pop_control_freq: Frequency at which population control occurs.
        rng_seed: The random number seed. If run in parallel the seeds on other
            cores / threads are determined from this.
    """
    # Due to structure of FT algorithm, `num_steps_per_block` is fixed at 1.
    # Overide whatever input for backward compatibility.
    num_steps_per_block: ClassVar[float] = 1 
    beta: float = _no_default
    pop_control_method: str = 'pair_branch'
    
    # This is a hack to get around the error:
    #
    #   TypeError: non-default argument 'beta' follows default argument
    #
    # due to inheritance from the QMCParams dataclass which has default attributes.
    # Ref: https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    def __post_init__(self):
        if self.beta is _no_default:
            raise TypeError("__init__ missing 1 required argument: 'beta'")

