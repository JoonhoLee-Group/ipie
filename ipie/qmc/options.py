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
from typing import Optional

from ipie.utils.io import get_input_value


def convert_from_reduced_unit(system, qmc_opts, verbose=False):
    if system.name == "UEG":
        TF = system.ef  # Fermi temeprature
        if verbose:
            print(f"# Fermi Temperature: {TF:13.8e}")
            print(f"# beta in reduced unit: {qmc_opts['beta']:13.8e}")
            print(f"# dt in reduced unit: {qmc_opts['dt']:13.8e}")
        dt = qmc_opts["dt"]  # original dt
        beta = qmc_opts["beta"]  # original dt
        scaled_dt = dt / TF  # converting to Hartree ^ -1
        scaled_beta = beta / TF  # converting to Hartree ^ -1
        if verbose:
            print(f"# beta in Hartree^-1:  {scaled_beta:13.8e}")
            print(f"# dt in Hartree^-1: {scaled_dt:13.8e}")
        return scaled_dt, scaled_beta


class QMCOpts(object):
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
    """

    # pylint: disable=dangerous-default-value
    # TODO: Remove this class / replace with dataclass
    def __init__(self, inputs={}, verbose=False):
        self.nwalkers = get_input_value(
            inputs, "num_walkers", default=None, alias=["nwalkers"], verbose=verbose
        )
        self.dt = get_input_value(inputs, "timestep", default=0.005, alias=["dt"], verbose=verbose)
        self.batched = get_input_value(inputs, "batched", default=True, verbose=verbose)
        self.nsteps = get_input_value(
            inputs, "num_steps", default=25, alias=["nsteps", "steps"], verbose=verbose
        )
        self.nblocks = get_input_value(
            inputs,
            "blocks",
            default=1000,
            alias=["num_blocks", "nblocks"],
            verbose=verbose,
        )
        self.nstblz = get_input_value(
            inputs,
            "stabilise_freq",
            default=5,
            alias=["nstabilise", "reortho"],
            verbose=verbose,
        )
        self.npop_control = get_input_value(
            inputs,
            "pop_control_freq",
            default=5,
            alias=["npop_control", "pop_control"],
            verbose=verbose,
        )
        self.pop_control_method = get_input_value(
            inputs,
            "pop_control_method",
            default="pair_branch",
            alias=["pop_control", "population_control"],
            verbose=verbose,
        )
        self.eqlb_time = get_input_value(
            inputs,
            "equilibration_time",
            default=2.0,
            alias=["tau_eqlb"],
            verbose=verbose,
        )
        self.neqlb = int(self.eqlb_time / self.dt)
        self.rng_seed = get_input_value(
            inputs,
            "rng_seed",
            default=None,
            alias=["random_seed", "seed"],
            verbose=verbose,
        )

    def __str__(self, verbose=0):
        _str = ""
        for k, v in self.__dict__.items():
            _str += f"# {k:<25s} : {v}\n"
        return _str


@dataclass
class QMCParams:
    r"""Input options and certain constants / parameters derived from them.

    Attributes
    ----------
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

    num_walkers: int
    total_num_walkers: int
    timestep: float
    num_steps_per_block: int
    num_blocks: int
    num_stblz: int = 5
    pop_control_freq: int = 5
    rng_seed: Optional[int] = None
