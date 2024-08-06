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

import numpy
import plum

from ipie.addons.free_projection.walkers.uhf_walkers import UHFWalkersFP, UHFWalkersParticleHoleFP
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler


@plum.dispatch
def UHFWalkersTrialFP(
    trial: SingleDet,
    initial_walker: numpy.ndarray,
    nup: int,
    ndown: int,
    nbasis: int,
    nwalkers: int,
    mpi_handler: MPIHandler,
    verbose: bool = False,
):
    return UHFWalkersFP(initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)


@plum.dispatch
def UHFWalkersTrialFP(
    trial: ParticleHole,
    initial_walker: numpy.ndarray,
    nup: int,
    ndown: int,
    nbasis: int,
    nwalkers: int,
    mpi_handler: MPIHandler,
    verbose: bool = False,
):
    return UHFWalkersParticleHoleFP(
        initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose
    )
