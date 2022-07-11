"""Routines for performing propagation of a walker"""
import sys

from ipie.propagation.continuous import Continuous


# TODO: Fix for discrete transformation.
def get_propagator_driver(system, hamiltonian, trial, qmc, options={}, verbose=False):
    hs = options.get("hubbard_stratonovich", "continuous")
    assert not ("discrete" in hs)
    return Continuous(system, hamiltonian, trial, qmc, options=options, verbose=verbose)
