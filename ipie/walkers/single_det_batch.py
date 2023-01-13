
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
import scipy.linalg

from ipie.estimators.greens_function_batch import greens_function
from ipie.propagation.overlap import calc_overlap_single_det, get_calc_overlap
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device
from ipie.utils.linalg import sherman_morrison
from ipie.utils.misc import get_numeric_names
from ipie.walkers.walker_batch import WalkerBatch


class SingleDetWalkerBatch(WalkerBatch):
    """UHF style walker.

    Parameters
    ----------
    system : object
        System object.
    hamiltonian : object
        Hamiltonian object.
    trial : object
        Trial wavefunction object.
    nwalkers : int
        The number of walkers in this batch
    walker_opts : dict
        Input options
    index : int
        Element of trial wavefunction to initalise walker to.
    nprop_tot : int
        Number of back propagation steps (including imaginary time correlation
                functions.)
    nbp : int
        Number of back propagation steps.
    """

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        nwalkers,
        walker_opts={},
        index=0,
        nprop_tot=None,
        nbp=None,
        mpi_handler=None,
    ):
        WalkerBatch.__init__(
            self,
            system,
            hamiltonian,
            trial,
            nwalkers,
            walker_opts=walker_opts,
            index=index,
            nprop_tot=nprop_tot,
            nbp=nbp,
            mpi_handler=mpi_handler,
        )

        self.name = "SingleDetWalkerBatch"

        calc_overlap = get_calc_overlap(trial)
        # self.ot = calc_overlap(self, trial)
        self.le_oratio = 1.0

        self.Ga = numpy.zeros(
            shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        if self.rhf:
            self.Gb = None
        else:
            self.Gb = numpy.zeros(
                shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )

        self.Ghalfa = numpy.zeros(
            shape=(nwalkers, system.nup, hamiltonian.nbasis), dtype=numpy.complex128
        )
        if self.rhf:
            self.Ghalfb = None
        else:
            self.Ghalfb = numpy.zeros(
                shape=(nwalkers, system.ndown, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )

        # greens_function(self, trial)

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        # WalkerBatch.cast_to_cupy(self, verbose)
        cast_to_device(self, verbose)
