import ast
import h5py
import numpy
import sys
import scipy.linalg
import time
from scipy.sparse import csr_matrix
from pie.utils.linalg import modified_cholesky
from pie.utils.io import (
        from_qmcpack_sparse,
        from_qmcpack_dense,
        write_qmcpack_sparse,
        write_qmcpack_dense,
        )
from pie.estimators.generic import (
        local_energy_generic, core_contribution,
        local_energy_generic_cholesky, core_contribution_cholesky
)
from pie.utils.mpi import get_shared_array, have_shared_mem


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

    def __init__(self, nelec, options = None, verbose=False):
        if verbose:
            print("# Parsing input options for systems.Generic.")
        self.name = "Generic"
        self.verbose = verbose
        self.nup, self.ndown = nelec
        self.nelec = nelec
        self.ne = self.nup + self.ndown
        
        if verbose:
            print("# Number of alpha electrons: %d"%(self.nup))
            print("# Number of beta electrons: %d"%(self.ndown))

        self.ktwist = numpy.array([None])