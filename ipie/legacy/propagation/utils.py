"""Routines for performing propagation of a walker"""
import sys

from ipie.legacy.propagation.continuous import Continuous
from ipie.legacy.propagation.hubbard import Hirsch
from ipie.legacy.propagation.hubbard_holstein import HirschDMC


# TODO: Fix for discrete transformation.
def get_propagator_driver(system, hamiltonian, trial, qmc, options={}, verbose=False):
    hs = options.get("hubbard_stratonovich", "continuous")

    if "discrete" in hs:
        return get_discrete_propagator(
            options, qmc, system, hamiltonian, trial, verbose
        )
    else:
        return Continuous(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )


def get_discrete_propagator(options, qmc, system, hamiltonian, trial, verbose=False):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.QMCOpts` class
        Trial wavefunction input options.
    hamiltonian : class
        hamiltonian class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    hs_type = options.get("hubbard_stratonovich", "discrete")
    if hamiltonian.name == "Hubbard":
        propagator = Hirsch(hamiltonian, trial, qmc, options=options, verbose=verbose)
    elif hamiltonian.name == "HubbardHolstein":
        propagator = HirschDMC(
            hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    else:
        print(
            "No suitable discrete propagator exists for {}. Check your input.".format(
                hamiltonian.name
            )
        )
        sys.exit()

    return propagator
