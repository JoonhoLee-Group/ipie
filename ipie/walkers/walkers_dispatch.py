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

from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import ParticleHoleWicks, ParticleHoleWicksSlow, ParticleHoleNaive, ParticleHoleWicksNonChunked
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase

import plum
from ipie.walkers.uhf_walkers import UHFWalkers, UHFWalkersParticleHole, UHFWalkersParticleHoleNaive
from ipie.walkers.ghf_walkers import GHFWalkers

def get_initial_walker(trial: TrialWavefunctionBase)->numpy.ndarray:
    if isinstance(trial, SingleDet):
        initial_walker = trial.psi
        num_dets = 1
    elif isinstance(trial, ParticleHoleWicks):
        initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
        num_dets = trial.num_dets
    elif isinstance(trial, ParticleHoleWicksNonChunked):
        initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
        num_dets = trial.num_dets
    else:
        raise Exception("Unrecognized trial type in get_initial_walker")
    return num_dets, initial_walker

# walker dispatcher based on trial type
@plum.dispatch
def GHFWalkersTrial (trial:SingleDet , initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return GHFWalkers (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)


@plum.dispatch
def UHFWalkersTrial (trial:SingleDet , initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkers (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)

@plum.dispatch
def UHFWalkersTrial (trial:ParticleHoleWicks, 
        initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkersParticleHole (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)

@plum.dispatch
def UHFWalkersTrial (trial:ParticleHoleWicksNonChunked, 
        initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkersParticleHole (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)

@plum.dispatch
def UHFWalkersTrial (trial:ParticleHoleWicksSlow, initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkersParticleHoleNaive (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)

@plum.dispatch
def UHFWalkersTrial (trial:ParticleHoleNaive, initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkersParticleHoleNaive (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)

@plum.dispatch
def UHFWalkersTrial (trial:NOCI, initial_walker:numpy.ndarray,
        nup:int, ndown:int, nbasis:int,
        nwalkers:int,
        mpi_handler=None,
        verbose:bool=False):
    return UHFWalkersParticleHoleNaive (initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose)











