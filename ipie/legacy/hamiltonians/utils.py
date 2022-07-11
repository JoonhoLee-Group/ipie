import sys
import time

import numpy

from ipie.hamiltonians.generic import construct_h1e_mod, read_integrals
from ipie.hamiltonians.utils import get_generic_integrals
from ipie.legacy.hamiltonians.generic import Generic
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.hamiltonians.ueg import UEG
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
    if ham_opts["name"] == "Hubbard":
        ham = Hubbard(ham_opts, verbose)
    elif ham_opts["name"] == "UEG":
        ham = UEG(system, ham_opts, verbose)
    elif ham_opts["name"] == "Generic":
        filename = ham_opts.get("integrals", None)
        if filename is None:
            if comm.rank == 0:
                print("# Error: integrals not specfied.")
                sys.exit()
        start = time.time()
        hcore, chol, h1e_mod, enuc = get_generic_integrals(
            filename, comm=comm, verbose=verbose
        )
        if verbose:
            print("# Time to read integrals: {:.6f}".format(time.time() - start))
        ham = Generic(
            h1e=hcore,
            chol=chol,
            ecore=enuc,
            h1e_mod=h1e_mod,
            options=ham_opts,
            verbose=verbose,
        )
    else:
        if comm.rank == 0:
            print("# Error: unrecognized hamiltonian name {}.".format(ham_opts["name"]))
            sys.exit()
        ham = None

    return ham
