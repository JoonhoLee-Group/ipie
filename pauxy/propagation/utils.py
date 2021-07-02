"""Routines for performing propagation of a walker"""
import sys
from pauxy.propagation.continuous import Continuous
from pauxy.propagation.hubbard import Hirsch
from pauxy.propagation.hubbard_holstein import HirschDMC

# TODO: Fix for discrete transformation.
def get_propagator_driver(system, trial, qmc, options={}, verbose=False):
    hs = options.get('hubbard_stratonovich', 'continuous')
    if 'discrete' in hs:
        return get_discrete_propagator(options, qmc, system, trial, verbose)
    else:
        return Continuous(system, trial, qmc, options=options, verbose=verbose)

def get_discrete_propagator(options, qmc, system, trial, verbose=False):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pauxy.qmc.QMCOpts` class
        Trial wavefunction input options.
    system : class
        System class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    hs_type = options.get('hubbard_stratonovich', 'discrete')
    if system.name == "Hubbard":
        propagator = Hirsch(system, trial, qmc,
                                options=options, verbose=verbose)
    elif system.name == "HubbardHolstein":
        propagator = HirschDMC(system, trial, qmc,
                                options=options, verbose=verbose)
    else:
        print("No suitable discrete propagator exists for {}. Check your input.".format(system.name))
        sys.exit()

    return propagator
