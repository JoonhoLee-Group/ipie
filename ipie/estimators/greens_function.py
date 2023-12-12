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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#          linusjoonho <linusjoonho@gmail.com>
#

from ipie.estimators.greens_function_multi_det import (
    greens_function_multi_det_wicks_opt,
    greens_function_noci,
)
from ipie.estimators.greens_function_single_det import (
    greens_function_single_det,
    greens_function_single_det_batch,
)
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.misc import is_cupy


def compute_greens_function(walker_batch, trial):
    compute_gf = get_greens_function(trial)
    return compute_gf(walker_batch, trial)


# Later we will add walker kinds as an input too
def get_greens_function(trial):
    """Wrapper to select the compute_greens_function function.

    Parameters
    ----------
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """

    if isinstance(trial, SingleDet):
        if is_cupy(
            trial.psi
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            compute_greens_function = greens_function_single_det_batch
        else:
            compute_greens_function = greens_function_single_det
    elif isinstance(trial, NOCI):
        compute_greens_function = greens_function_noci
    elif isinstance(trial, ParticleHole):
        compute_greens_function = greens_function_multi_det_wicks_opt
    else:
        compute_greens_function = None

    return compute_greens_function


def greens_function(walker_batch, trial, build_full=False):
    compute_greens_function = get_greens_function(trial)
    return compute_greens_function(walker_batch, trial, build_full=build_full)
