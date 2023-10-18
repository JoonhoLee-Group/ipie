import math
import time

import numpy
from ipie.config import config
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_base import GenericBase
from ipie.thermal.propagation.operations import apply_exponential
from ipie.thermal.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers


class PhaselessGeneric(PhaselessBase):
    """A class for performing phaseless propagation with real, generic, hamiltonian."""

    def __init__(self, time_step, mu, exp_nmax=6, lowrank=False, verbose=False):
        super().__init__(time_step, mu, lowrank=lowrank, verbose=verbose)
        self.exp_nmax = exp_nmax

    def construct_VHS(self, hamiltonian, xshifted):
        """Includes `nwalkers`.
        """
        nwalkers = xshifted.shape[-1] # Shape (nfields, nwalkers).
        VHS = hamiltonian.chol.dot(xshifted) # Shape (nbasis^2, nwalkers).
        VHS = self.isqrt_dt * VHS.T.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS # Shape (nwalkers, nbasis, nbasis).
