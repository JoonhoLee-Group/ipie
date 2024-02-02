import time
import plum
import math
import cmath
import numpy
import scipy.linalg

from abc import abstractmethod
from ipie.propagation.continuous_base import ContinuousBase
from ipie.thermal.estimators.thermal import one_rdm_from_G
from ipie.thermal.propagation.force_bias import construct_force_bias
from ipie.thermal.propagation.operations import apply_exponential
from ipie.utils.backend import arraylib as xp
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol

# TODO: Add lowrank implementation.
# Ref: 10.1103/PhysRevB.80.214116 for bounds.

@plum.dispatch
def construct_mean_field_shift(hamiltonian: GenericRealChol, trial):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} P_{ik\sigma}

    """
    # hamiltonian.chol has shape (nbasis^2, nchol).
    P = one_rdm_from_G(trial.G)
    P = (P[0] + P[1]).ravel()
    tmp_real = numpy.dot(hamiltonian.chol.T, P.real)
    tmp_imag = numpy.dot(hamiltonian.chol.T, P.imag)
    mf_shift = 1.0j * tmp_real - tmp_imag
    return mf_shift # Shape (nchol,).


@plum.dispatch
def construct_mean_field_shift(hamiltonian: GenericComplexChol, trial):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} P_{ik\sigma}

    """
    # hamiltonian.chol has shape (nbasis^2, nchol).
    P = one_rdm_from_G(trial.G)
    P = (P[0] + P[1]).ravel()
    nchol = hamiltonian.nchol
    mf_shift = numpy.zeros(hamiltonian.nfields, dtype=hamiltonian.chol.dtype)
    mf_shift[:nchol] = 1j * numpy.dot(hamiltonian.A.T, P.ravel())
    mf_shift[nchol:] = 1j * numpy.dot(hamiltonian.B.T, P.ravel())
    return mf_shift # Shape (nchol,).


class PhaselessBase(ContinuousBase):
    """A base class for generic continuous HS transform FT-AFQMC propagators."""  

    def __init__(self, time_step, mu, lowrank=False, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.mu = mu
        self.sqrt_dt = self.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt

        self.nfb_trig = 0  # number of force bias triggered
        self.ebound = (2.0 / self.dt) ** 0.5  # energy bound range
        self.fbbound = 1.0
        self.mpi_handler = None
        self.lowrank = lowrank


    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        # dt/2 one-body propagator
        start = time.time()
        self.mf_shift = construct_mean_field_shift(hamiltonian, trial)

        if verbose:
            print(f"# Time to mean field shift: {time.time() - start} s")
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )

        # Construct one-body propagator
        self.BH1 = self.construct_one_body_propagator(hamiltonian)

        # Allocate force bias (we don't need to do this here - it will be allocated when it is needed)
        self.vbias = None

        # Legacy attributes.
        self.mf_core = hamiltonian.ecore + 0.5 * numpy.dot(self.mf_shift, self.mf_shift)
        self.mf_const_fac = cmath.exp(-self.dt * self.mf_core)


    @plum.dispatch
    def construct_one_body_propagator(self, hamiltonian: GenericRealChol):
        r"""Construct mean-field shifted one-body propagator.

        .. math::

            H1 \rightarrow H1 - v0
            v0_{ik} = \sum_n v_{(ik),n} \bar{v}_n

        Parameters
        ----------
        hamiltonian : hamiltonian class
            Generic hamiltonian object.
        dt : float
            Timestep.
        """
        nb = hamiltonian.nbasis
        shift = 1j * numpy.einsum("mx,x->m", hamiltonian.chol, self.mf_shift).reshape(nb, nb)
        muN = self.mu * numpy.identity(nb, dtype=hamiltonian.H1.dtype)
        H1 = hamiltonian.h1e_mod - numpy.array([shift + muN, shift + muN])
        expH1 = numpy.array([
                    scipy.linalg.expm(-0.5 * self.dt * H1[0]), 
                    scipy.linalg.expm(-0.5 * self.dt * H1[1])])
        return expH1 # Shape (nbasis, nbasis).

    
    @plum.dispatch
    def construct_one_body_propagator(self, hamiltonian: GenericComplexChol):
        r"""Construct mean-field shifted one-body propagator.

        .. math::

            H1 \rightarrow H1 - v0
            v0_{ik} = \sum_n v_{(ik),n} \bar{v}_n

        Parameters
        ----------
        hamiltonian : hamiltonian class
            Generic hamiltonian object.
        dt : float
            Timestep.
        """
        nb = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        shift = xp.zeros((nb, nb), dtype=hamiltonian.chol.dtype)
        shift = 1j * numpy.einsum("mx,x->m", hamiltonian.A, self.mf_shift[:nchol]).reshape(nb, nb)
        shift += 1j * numpy.einsum("mx,x->m", hamiltonian.B, self.mf_shift[nchol:]).reshape(nb, nb)
        muN = self.mu * numpy.identity(nb, dtype=hamiltonian.H1.dtype)
        H1 = hamiltonian.h1e_mod - numpy.array([shift + muN, shift + muN])
        expH1 = numpy.array([
                    scipy.linalg.expm(-0.5 * self.dt * H1[0]), 
                    scipy.linalg.expm(-0.5 * self.dt * H1[1])])
        return expH1 # Shape (nbasis, nbasis).


    def construct_two_body_propagator(self, walkers, hamiltonian, trial, debug=False):
        r"""Construct two-body propagator.

        .. math::
            \bar{x}_n &= \sqrt{\Delta\tau} \bar{v}_n \\
            x_\mathrm{shifted}_n &= x_n - \bar{x}_n \\
            C_{MF} &= -\sqrt{\Delta\tau} \sum_{n} x_\mathrm{shifted}_n \bar{v}_n \\
            &= -\sqrt{\Delta\tau} \sum_{n} (x_n - \sqrt{\Delta\tau} \bar{v}_n) \bar{v}_n \\
            &= -\sqrt{\Delta\tau} \sum_{n} x_n \bar{v}_n + \Delta\tau \sum_{n} \bar{v}_n^2.

        Parameters
        ----------
        walkers: walker class
            UHFThermalWalkers object.
        hamiltonian : hamiltonian class
            Generic hamiltonian object.
        trial : trial class
            Trial dnsity matrix.
        """
        # Optimal force bias
        xbar = xp.zeros((walkers.nwalkers, hamiltonian.nfields)) 
        start_time = time.time()
        self.vbias = construct_force_bias(hamiltonian, walkers)
        xbar = -self.sqrt_dt * (1j * self.vbias - self.mf_shift)
        self.timer.tfbias += time.time() - start_time

        # Force bias bounding
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)

        # Normally distrubted auxiliary fields.
        xi = xp.random.normal(0.0, 1.0, hamiltonian.nfields * walkers.nwalkers).reshape(
                walkers.nwalkers, hamiltonian.nfields)

        if debug: self.xi = xi # For debugging.
        xshifted = xi - xbar # Shape (nwalkers, nfields).
        
        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted, self.mf_shift) # Shape (nwalkers,).
        # Constant factor arising from shifting the propability distribution.
        cfb = xp.einsum("wx,wx->w", xi, xbar) - 0.5 * xp.einsum("wx,wx->w", xbar, xbar) # Shape (nwalkers,).

        xshifted = xshifted.T.copy() # Shape (nfields, nwalkers).
        VHS = self.construct_VHS(hamiltonian, xshifted) # Shape (nwalkers, nbasis, nbasis).
        return cmf, cfb, xshifted, VHS

    def propagate_walkers_one_body(self, walkers):
        pass

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        pass
    
    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=0., debug=False):
        start_time = time.time()
        cmf, cfb, xshifted, VHS = self.construct_two_body_propagator(
                                    walkers, hamiltonian, trial, debug=debug)
        assert walkers.nwalkers == xshifted.shape[-1]
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3

        start_time = time.time()
        for iw in range(walkers.nwalkers):
            stack = walkers.stack[iw]
            BV = apply_exponential(VHS[iw], self.exp_nmax) # Shape (nbasis, nbasis).
            B = numpy.array([BV.dot(self.BH1[0]), BV.dot(self.BH1[1])])
            B = numpy.array([self.BH1[0].dot(B[0]), self.BH1[1].dot(B[1])])

            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's Green's function.
            tix = stack.nslice
            start_time = time.time()
            G = walkers.calc_greens_function(iw, slice_ix=tix, inplace=False)
            self.timer.tgf += time.time() - start_time

            # 2. Compute updated Green's function.
            start_time = time.time()
            stack.update_new(B)
            walkers.calc_greens_function(iw, slice_ix=tix, inplace=True)

            # 3. Compute det(G/G')
            # Now apply phaseless approximation
            if debug:
                self.update_weight_legacy(walkers, iw, G, cfb, cmf, eshift)

            else:
                self.update_weight(walkers, iw, G, cfb, cmf, eshift)

            self.timer.tupdate += time.time() - start_time


    def update_weight(self, walkers, iw, G, cfb, cmf, eshift):
        """Update weight for walker `iw`.
        """
        M0a = scipy.linalg.det(G[0], check_finite=False)
        M0b = scipy.linalg.det(G[1], check_finite=False)
        Mnewa = scipy.linalg.det(walkers.Ga[iw], check_finite=False)
        Mnewb = scipy.linalg.det(walkers.Gb[iw], check_finite=False)
        
        # ovlp = det( G^{-1} )
        ovlp_ratio = (M0a * M0b) / (Mnewa * Mnewb) # ovlp_new / ovlp_old
        hybrid_energy = -(xp.log(ovlp_ratio) + cfb[iw] + cmf[iw]) / self.dt # Scalar.
        hybrid_energy = self.apply_bound_hybrid(hybrid_energy, eshift)
        importance_function = xp.exp(
            -self.dt * (0.5 * (hybrid_energy + walkers.hybrid_energy) - eshift))

        # Splitting w_k = |I(x, \bar{x}, |phi_k>)| e^{i theta_k}, where `k` 
        # labels the time slice.
        magn = xp.abs(importance_function)
        walkers.hybrid_energy = hybrid_energy
        dtheta = (-self.dt * hybrid_energy - cfb[iw]).imag # Scalar.
        cosine_fac = xp.amax([0., xp.cos(dtheta)])
        walkers.weight[iw] *= magn * cosine_fac
        walkers.M0a[iw] = Mnewa
        walkers.M0b[iw] = Mnewb


    def update_weight_legacy(self, walkers, iw, G, cfb, cmf, eshift):
        """Update weight for walker `iw` using legacy code.
        """
        M0a = scipy.linalg.det(G[0], check_finite=False)
        M0b = scipy.linalg.det(G[1], check_finite=False)
        Mnewa = scipy.linalg.det(walkers.Ga[iw], check_finite=False)
        Mnewb = scipy.linalg.det(walkers.Gb[iw], check_finite=False)
        _cfb = cfb[iw]
        _cmf = cmf[iw]

        try:
            # Could save M0 rather than recompute.
            oratio = (M0a * M0b) / (Mnewa * Mnewb)
            # Might want to cap this at some point
            hybrid_energy = cmath.log(oratio) + _cfb + _cmf
            Q = cmath.exp(hybrid_energy)
            #hybrid_energy = -(cmath.log(oratio) + _cfb + _cmf) / self.dt
            #walkers.hybrid_energy = hybrid_energy + self.mf_core
            #Q = cmath.exp(-self.dt * hybrid_energy)
            #hybrid_energy = cmath.log(oratio) + _cfb + _cmf
            expQ = self.mf_const_fac * Q
            (magn, phase) = cmath.polar(expQ)

            if not math.isinf(magn):
                # Determine cosine phase from Arg(det(1+A'(x))/det(1+A(x))).
                # Note this doesn't include exponential factor from shifting
                # proability distribution.
                dtheta = cmath.phase(cmath.exp(hybrid_energy - _cfb))
                cosine_fac = max(0, math.cos(dtheta))
                walkers.weight[iw] *= magn * cosine_fac
                walkers.M0a[iw] = Mnewa
                walkers.M0b[iw] = Mnewb

            else:
                walkers.weight[iw] = 0.

        except ZeroDivisionError:
            walkers.weight[iw] = 0.

    def apply_bound_force_bias(self, xbar, max_bound=1.0):
        absxbar = xp.abs(xbar)
        idx_to_rescale = absxbar > max_bound
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.copy()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        xbar = xp.where(idx_to_rescale, xbar_rescaled, xbar)
        self.nfb_trig += xp.sum(idx_to_rescale)
        return xbar


    def apply_bound_hybrid(self, ehyb, eshift):  # Shift is a number but ehyb is not
        # For initial steps until first estimator communication, `eshift` will be
        # zero and hybrid energy can be incorrect. So just avoid capping for
        # first block until reasonable estimate of `eshift` can be computed.
        if abs(eshift) < 1e-10:
            return ehyb

        emax = eshift.real + self.ebound
        emin = eshift.real - self.ebound
        return xp.minimum(emax, xp.maximum(ehyb, emin))


    # Form VHS.
    @abstractmethod
    def construct_VHS(self, hamiltonian, xshifted):
        pass

