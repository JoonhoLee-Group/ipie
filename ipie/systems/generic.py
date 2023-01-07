
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
#

import ast
import sys
import time

import h5py
import numpy
import scipy.linalg
from scipy.sparse import csr_matrix

from ipie.utils.io import (from_qmcpack_dense, from_qmcpack_sparse,
                           write_qmcpack_dense, write_qmcpack_sparse)
from ipie.utils.linalg import modified_cholesky
from ipie.utils.mpi import get_shared_array, have_shared_mem


class Generic(object):
    """Generic system class

    This class should contain information that is system specific and not related to the hamiltonian

    Parameters
    ----------
    nelec : tuple
        Number of alpha and beta electrons.
    inputs : dict
        Input options defined below.
    verbose : bool
        Print extra information.

    Attributes
    ----------
    nup : int
        number of alpha electrons
    ndown : int
        number of beta electrons
    ne : int
        total number of electrons
    nelec : tuple
        Number of alpha and beta electrons.
    """

    def __init__(self, nelec, options=None, verbose=False):
        if verbose:
            print("# Parsing input options for systems.Generic.")
        self.name = "Generic"
        self.verbose = verbose
        self.nup, self.ndown = nelec
        self.nelec = nelec
        self.ne = self.nup + self.ndown

        if verbose:
            print("# Number of alpha electrons: %d" % (self.nup))
            print("# Number of beta electrons: %d" % (self.ndown))

        self.ktwist = numpy.array([None])
