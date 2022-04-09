import numpy
import sys
import time

from ipie.hamiltonians.generic import Generic, read_integrals, construct_h1e_mod
from ipie.legacy.hamiltonians.ueg import UEG
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.utils.mpi import get_shared_array, have_shared_mem

def get_hamiltonian(system, ham_opts=None, verbose=0, comm=None):
    """Wrapper to select hamiltonian class

    Parameters
    ----------
    ham_opts : dict
        Hamiltonian input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    ham : object
        Hamiltonian class.
    """
    if ham_opts['name'] == 'Hubbard':
        ham = Hubbard(ham_opts, verbose)
    elif ham_opts['name'] == 'UEG':
        ham = UEG(system, ham_opts, verbose)
    else:
        import ipie.hamiltonians.utils
        ham = ipie.hamiltonians.utils.get_hamiltonian(system, ham_opts, verbose, comm)
        # if comm.rank == 0:
        #     print("# Error: unrecognized hamiltonian name {}.".format(ham_opts['name']))
        #     sys.exit()
        # ham = None

    return ham
