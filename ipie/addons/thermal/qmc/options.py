# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

from dataclasses import dataclass
from typing import ClassVar, Optional

from ipie.utils.io import get_input_value
from ipie.qmc.options import QMCOpts, QMCParams


class ThermalQMCOpts(QMCOpts):
    r"""Input options and certain constants / parameters derived from them.

    Initialised from a dict containing the following options, not all of which
    are required.

    Attributes
    ----------
    batched : bool
        Whether to do batched calculations.
    nwalkers : int
        Number of walkers to propagate in a simulation.
    dt : float
        Timestep.
    nsteps : int
        Number of steps per block.
    nblocks : int
        Number of blocks. Total number of iterations = nblocks * nsteps.
    nstblz : int
        Frequency of Gram-Schmidt orthogonalisation steps.
    npop_control : int
        Frequency of population control.
    pop_control_method : str
        Population control method.
    eqlb_time : float
        Time scale of equilibration phase. Only used to fix local
        energy bound when using phaseless approximation.
    neqlb : int
        Number of time steps for the equilibration phase. Only used to fix the
        local energy bound when using phaseless approximation.
    rng_seed : int
        The random number seed.
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.
    """
    # pylint: disable=dangerous-default-value
    # TODO: Remove this class / replace with dataclass
    def __init__(self, inputs={}, verbose=False):
        super().__init__(inputs, verbose)

        self.mu = get_input_value(
            inputs,
            "mu",
            default=None,
            verbose=verbose,
        )
        self.beta = get_input_value(
            inputs,
            "beta",
            default=None,
            verbose=verbose,
        )


@dataclass
class ThermalQMCParams(QMCParams):
    r"""Input options and certain constants / parameters derived from them.

    Attributes
    ----------
    mu : float
        Chemical potential.
    beta : float 
        Inverse temperature.
    num_walkers : int
        Number of walkers **per** core / task / computational unit.
    total_num_walkers : int
        The total number of walkers in the simulation.
    timestep : float
        The timestep delta_t
    num_steps_per_block : int
        Number of steps of propagation before estimators are evaluated.
    num_blocks : int
        Number of blocks. Total number of iterations = num_blocks * num_steps_per_block.
    num_stblz : int
        Number of steps before QR stabilization of walkers is performed.
    pop_control_freq : int
        Frequency at which population control occurs.
    rng_seed : int
        The random number seed. If run in parallel the seeds on other cores / 
        threads are determined from this.
    """
    # Due to structure of FT algorithm, `num_steps_per_block` is fixed at 1.
    # Overide whatever input for backward compatibility.
    num_steps_per_block: ClassVar[int] = 1 
    mu: Optional[float] = None
    beta: Optional[float] = None
    pop_control_method: str = 'pair_branch'
    
    def __post_init__(self):
        if self.mu is None:
            raise TypeError("__init__ missing 1 required argument: 'mu'")
        if self.beta is None:
            raise TypeError("__init__ missing 1 required argument: 'beta'")

