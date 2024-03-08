import time
import math
import plum

import numpy
from ipie.config import config
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.generic_base import GenericBase
from ipie.addons.thermal.propagation.operations import apply_exponential
from ipie.addons.thermal.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers


class PhaselessGeneric(PhaselessBase):
    """A class for performing phaseless propagation with real, generic, hamiltonian."""

    def __init__(self, time_step, mu, exp_nmax=6, lowrank=False, verbose=False):
        super().__init__(time_step, mu, lowrank=lowrank, verbose=verbose)
        self.exp_nmax = exp_nmax
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericRealChol, xshifted: xp.ndarray):
        """Includes `nwalkers`.
        """
        nwalkers = xshifted.shape[-1] # Shape (nfields, nwalkers).
        VHS = hamiltonian.chol.dot(xshifted) # Shape (nbasis^2, nwalkers).
        VHS = self.isqrt_dt * VHS.T.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS # Shape (nwalkers, nbasis, nbasis).
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericComplexChol, xshifted: xp.ndarray):
        """Includes `nwalkers`.
        """
        nwalkers = xshifted.shape[-1]
        nchol = hamiltonian.nchol
        VHS = self.isqrt_dt * (
            hamiltonian.A.dot(xshifted[:nchol]) + hamiltonian.B.dot(xshifted[nchol:])
        )
        VHS = VHS.T.copy()
        VHS = VHS.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS
