import sys

import numpy

from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.systems.hubbard_holstein import HubbardHolstein
from ipie.legacy.systems.ueg import UEG
from ipie.systems.generic import Generic
from ipie.utils.mpi import get_shared_array, have_shared_mem


def get_system(sys_opts=None, verbose=0, comm=None):
    """Wrapper to select system class

    Parameters
    ----------
    sys_opts : dict
        System input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    system : object
        System class.
    """
    if not ("name" in sys_opts):
        sys_opts["name"] = "Generic"

    if sys_opts["name"] == "UEG":
        system = UEG(sys_opts, verbose)
    elif (
        sys_opts["name"] == "Hubbard"
        or sys_opts["name"] == "HubbardHolstein"
        or sys_opts["name"] == "Generic"
    ):
        nup, ndown = sys_opts.get("nup"), sys_opts.get("ndown")
        if nup is None or ndown is None:
            if comm.rank == 0:
                print("# Error: Number of electrons not specified.")
                sys.exit()
        nelec = (nup, ndown)
        system = Generic(nelec, sys_opts, verbose)
    else:
        if comm.rank == 0:
            print("# Error: unrecognized system name {}.".format(sys_opts["name"]))
            sys.exit()

    return system
