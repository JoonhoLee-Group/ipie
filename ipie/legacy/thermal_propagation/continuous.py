import cmath
import math
import sys
import time

import numpy
import scipy.sparse.linalg

from ipie.legacy.estimators.thermal import (inverse_greens_function_qr,
                                            one_rdm_from_G)
from ipie.legacy.thermal_propagation.generic import GenericContinuous
from ipie.legacy.thermal_propagation.hubbard import HubbardContinuous
from ipie.legacy.thermal_propagation.planewave import PlaneWave
from ipie.propagation.operations import kinetic_real
from ipie.utils.linalg import exponentiate_matrix


class Continuous(object):
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

    def __init__(
        self, options, qmc, system, hamiltonian, trial, verbose=False, lowrank=False
    ):
        if verbose:
            print("# Parsing continuous propagator input options.")
            print("# Using continuous Hubbar--Stratonovich transformations.")

        # Input options
        self.hs_type = "continuous"
        self.exp_nmax = options.get("expansion_order", 6)

        optimised = options.get("optimised", True)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt
        self.nfb_trig = 0
        self.lowrank = lowrank

        self.propagator = get_continuous_propagator(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
        # Mean field shifted one-body propagator
        self.mu = system.mu
        self.propagator.construct_one_body_propagator(hamiltonian, qmc.dt)

        self.BH1 = self.propagator.BH1
        self.BT = trial.dmat
        self.BTinv = trial.dmat_inv
        self.BT_BP = None

        self.mf_const_fac = cmath.exp(-self.dt * self.propagator.mf_core)
        self.nstblz = qmc.nstblz

        self.ebound = (2.0 / self.dt) ** 0.5
        self.mean_local_energy = 0
        self.free_projection = options.get("free_projection", False)
        self.force_bias = options.get("force_bias", True)
        if self.free_projection:
            if verbose:
                print("# Using free projection.")
                print("# Setting force_bias to False with free projection.")
            self.force_bias = options.get("force_bias", False)
            self.propagate_walker = self.propagate_walker_free
        else:
            if verbose:
                print("# Using phaseless approximation.")
                if self.force_bias:
                    print("# Setting force bias to %r." % self.force_bias)
            self.propagate_walker = self.propagate_walker_phaseless
        if verbose:
            print("# Finished setting up propagator.")

    def two_body_propagator(self, walker, system, trial):
        r"""Continuous Hubbard-Statonovich transformation.

        Parameters
        ----------
        walker : :class:`pie.walker.Walker` walker object to be updated. On
            output we have acted on phi by B_V(x).
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, system.nfields)
        if self.force_bias:
            P = one_rdm_from_G(walker.G)
            xbar = self.propagator.construct_force_bias(system, P, trial)
        else:
            xbar = numpy.zeros(xi.shape, dtype=numpy.complex128)

        for i in range(system.nfields):
            if numpy.absolute(xbar[i]) > 1.0:
                # if self.nfb_trig < 10:
                # print("# Rescaling force bias is triggered")
                # print("# Warning will only be printed 10 times on root.")
                self.nfb_trig += 1
                xbar[i] /= numpy.absolute(xbar[i])
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5 * xbar.dot(xbar)
        xshifted = xi - xbar
        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xshifted.dot(self.propagator.mf_shift)

        # Operator terms contributing to propagator.
        VHS = self.propagator.construct_VHS(system, xshifted)

        return (cmf, cfb, xshifted, VHS)

    def estimate_eshift(self, walker):
        return 0.0

    def exponentiate(self, VHS, debug=False):
        """Apply exponential propagator of the HS transformation
        Parameters
        ----------
        system :
            system class
        phi : numpy array
            a state
        VHS : numpy array
            HS transformation potential
        Returns
        -------
        phi : numpy array
            Exp(VHS) * phi
        """
        # JOONHO: exact exponential
        # copy = numpy.copy(phi)
        # phi = scipy.linalg.expm(VHS).dot(copy)
        phi = numpy.identity(VHS.shape[0], dtype=numpy.complex128)
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        Temp = numpy.identity(VHS.shape[0], dtype=numpy.complex128)

        for n in range(1, self.exp_nmax + 1):
            Temp = VHS.dot(Temp) / n
            phi += Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

    def propagate_walker_free(self, system, walker, trial, eshift=0):
        r"""Free projection for continuous HS transformation.

        .. Warning::
            Currently not implemented.


        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. on output we have acted on
            :math:`|\phi_i\rangle` by :math:`B` and updated the weight
            appropriately. Updates inplace.
        state : :class:`state.State`
            Simulation state.
        """
        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, system, trial)
        BV = self.exponentiate(VHS)

        B = numpy.array([BV.dot(self.BH1[0]), BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]), self.BH1[1].dot(B[1])])

        # Compute determinant ratio det(1+A')/det(1+A).
        # 1. Current walker's green's function.
        G = walker.greens_function(trial, inplace=False)
        # 2. Compute updated green's function.
        walker.stack.update_new(B)
        walker.greens_function(trial, inplace=True)
        # 3. Compute det(G/G')
        M0 = [
            scipy.linalg.det(G[0], check_finite=False),
            scipy.linalg.det(G[1], check_finite=False),
        ]
        Mnew = [
            scipy.linalg.det(walker.G[0], check_finite=False),
            scipy.linalg.det(walker.G[1], check_finite=False),
        ]
        # Could save M0 rather than recompute.
        try:
            # Could save M0 rather than recompute.
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])
            walker.ot = 1.0
            # Constant terms are included in the walker's weight.
            (magn, phase) = cmath.polar(cmath.exp(cmf + cfb) * oratio)
            walker.weight *= magn
            walker.phase *= cmath.exp(1j * phase)
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_walker_phaseless(self, system, walker, trial, eshift=0):
        r"""Propagate walker using phaseless approximation.

        Uses importance sampling and the hybrid method.

        Parameters
        ----------
        walker : :class:`walker.Walker`
            Walker object to be updated. On output we have acted on phi with the
            propagator B(x), and updated the weight appropriately.  Updates
            inplace.
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, system, trial)
        BV = self.exponentiate(VHS)

        B = numpy.array([BV.dot(self.BH1[0]), BV.dot(self.BH1[1])])
        B = numpy.array([self.BH1[0].dot(B[0]), self.BH1[1].dot(B[1])])

        # Compute determinant ratio det(1+A')/det(1+A).
        # 1. Current walker's green's function.
        tix = walker.stack.ntime_slices
        # 2. Compute updated green's function.
        walker.stack.update_new(B)
        walker.greens_function(None, slice_ix=tix, inplace=True)
        # 3. Compute det(G/G')
        M0 = walker.M0
        Mnew = [
            scipy.linalg.det(walker.G[0], check_finite=False),
            scipy.linalg.det(walker.G[1], check_finite=False),
        ]
        try:
            # Could save M0 rather than recompute.
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])
            # Might want to cap this at some point
            hybrid_energy = cmath.log(oratio) + cfb + cmf
            Q = cmath.exp(hybrid_energy)
            expQ = self.mf_const_fac * Q
            (magn, phase) = cmath.polar(expQ)

            if not math.isinf(magn):
                # Determine cosine phase from Arg(det(1+A'(x))/det(1+A(x))).
                # Note this doesn't include exponential factor from shifting
                # propability distribution.
                dtheta = cmath.phase(cmath.exp(hybrid_energy - cfb))
                cosine_fac = max(0, math.cos(dtheta))
                walker.weight *= magn * cosine_fac
                walker.M0 = Mnew
            else:
                walker.weight = 0.0
        except ZeroDivisionError:
            walker.weight = 0.0


def get_continuous_propagator(
    system, hamiltonian, trial, qmc, options={}, verbose=False
):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.QMCOpts` class
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
    if system.name == "UEG":
        propagator = PlaneWave(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    elif system.name == "Hubbard":
        propagator = HubbardContinuous(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    elif system.name == "Generic":
        propagator = GenericContinuous(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    else:
        propagator = None

    return propagator
