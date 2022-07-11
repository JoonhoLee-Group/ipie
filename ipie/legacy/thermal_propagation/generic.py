import cmath
import math
import sys
import time

import numpy
import scipy.sparse.linalg

from ipie.legacy.estimators.thermal import (inverse_greens_function_qr,
                                            one_rdm_from_G)
from ipie.propagation.operations import kinetic_real
from ipie.utils.linalg import exponentiate_matrix


class GenericContinuous(object):
    """Propagator for generic many-electron Hamiltonian.

    Uses continuous HS transformation for exponential of two body operator.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.options.QMCOpts`
        QMC options.
    system : :class:`pie.system.System`
        System object.
    trial : :class:`pie.trial_wavefunctioin.Trial`
        Trial wavefunction object.
    verbose : bool
        If true print out more information during setup.
    """

    def __init__(self, system, hamiltonian, trial, qmc, options={}, verbose=False):
        if verbose:
            print("# Parsing continuous propagator input options.")

        # Input options
        self.hs_type = "continuous"
        self.free_projection = options.get("free_projection", False)
        if verbose:
            print("# Using phaseless approximation: %r" % (not self.free_projection))
        self.exp_nmax = options.get("expansion_order", 6)
        self.force_bias = options.get("force_bias", True)
        if self.free_projection:
            if verbose:
                print("# Setting force_bias to False with free projection.")
            self.force_bias = False
        else:
            if verbose:
                print("# Setting force bias to %r." % self.force_bias)

        optimised = options.get("optimised", True)
        if optimised:
            self.construct_force_bias = self.construct_force_bias_fast
            self.construct_VHS = self.construct_VHS_fast
        else:
            self.construct_force_bias = self.construct_force_bias_slow
            self.construct_VHS = self.construct_VHS_slow
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt
        self.nfb_trig = 0

        P = one_rdm_from_G(trial.G)
        # Mean field shifts (2,nchol_vec).
        self.mf_shift = self.construct_mean_field_shift(hamiltonian, P)
        if verbose:
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )

        # Mean field shifted one-body propagator
        self.mu = hamiltonian.mu

        self.BT = trial.dmat
        self.BTinv = trial.dmat_inv

        # Constant core contribution modified by mean field shift.
        self.mf_core = hamiltonian.ecore + 0.5 * numpy.dot(self.mf_shift, self.mf_shift)
        self.nstblz = qmc.nstblz

        self.ebound = (2.0 / self.dt) ** 0.5
        self.mean_local_energy = 0
        if verbose:
            print("# Finished setting up propagator.")

    def construct_mean_field_shift(self, system, P):
        if system.sparse:
            mf_shift = 1j * P[0].ravel() * system.chol_vecs
            mf_shift += 1j * P[1].ravel() * system.chol_vecs
        else:
            mf_shift = 1j * numpy.einsum("lpq,spq->l", system.chol_vecs, P)
        return mf_shift

    def construct_one_body_propagator(self, system, dt):
        """Construct mean-field shifted one-body propagator.

        Parameters
        ----------
        dt : float
            Timestep.
        chol_vecs : :class:`numpy.ndarray`
            Cholesky vectors.
        h1e_mod : :class:`numpy.ndarray`
            One-body operator including factor from factorising two-body
            Hamiltonian.
        """
        if system.sparse:
            nb = system.nbasis
            shift = 1j * system.hs_pot.dot(self.mf_shift).reshape(nb, nb)
        else:
            shift = 1j * numpy.einsum("l,lpq->pq", self.mf_shift, system.hs_pot)
        I = numpy.identity(system.nbasis, dtype=system.H1.dtype)
        muN = self.mu * I
        H1 = system.h1e_mod - numpy.array([shift + muN, shift + muN])

        self.BH1 = numpy.array(
            [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
        )

    def construct_force_bias_slow(self, system, P, trial):
        r"""Compute optimal force bias.

        Uses explicit expression.

        Parameters
        ----------
        G: :class:`numpy.ndarray`
            Walker's 1RDM: <c_i^{\dagger}c_j>.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        vbias = numpy.einsum("lpq,pq->l", system.hs_pot, P[0])
        vbias += numpy.einsum("lpq,pq->l", system.hs_pot, P[1])
        return -self.sqrt_dt * (1j * vbias - self.mf_shift)

    def construct_force_bias_fast(self, system, P, trial):
        r"""Compute optimal force bias.

        Uses explicit expression.

        Parameters
        ----------
        G: :class:`numpy.ndarray`
            Walker's 1RDM: <c_i^{\dagger}c_j>.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        vbias = P[0].ravel() * system.hs_pot
        vbias += P[1].ravel() * system.hs_pot
        return -self.sqrt_dt * (1j * vbias - self.mf_shift)

    def construct_VHS_slow(self, system, shifted):
        return self.isqrt_dt * numpy.einsum("l,lpq->pq", shifted, system.hs_pot)

    def construct_VHS_fast(self, system, xshifted):
        VHS = system.hs_pot.dot(xshifted)
        VHS = VHS.reshape(system.nbasis, system.nbasis)
        return self.isqrt_dt * VHS
