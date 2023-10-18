import time
import numpy
import scipy.linalg

from abc import abstractmethod
from ipie.propagation.continuous_base import ContinuousBase
from ipie.thermal.propagation.force_bias import construct_force_bias
from ipie.thermal.propagation.operations import apply_exponential
from ipie.utils.backend import arraylib as xp
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol

# TODO write test for propagator.

def construct_mean_field_shift(hamiltonian, trial):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    """
    # hamiltonian.chol has shape (nbasis^2, nchol).
    P = (trial.P[0] + trial.P[1]).ravel()
    tmp_real = numpy.dot(hamiltonian.chol.T, P.real)
    tmp_imag = numpy.dot(hamiltonian.chol.T, P.imag)
    mf_shift = 1.0j * tmp_real - tmp_imag
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


    def construct_one_body_propagator(self, hamiltonian):
        r"""Construct mean-field shifted one-body propagator.

        .. math::

            H1 \rightarrow H1 - v0
            v0_{ik} = \sum_n v_{(ik),n} \bar{v}_n

        Parameters
        ----------
        hamiltonian : hamiltonian class.
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


    def construct_two_body_propagator(self, walkers, hamiltonian, trial):
        """Includes `nwalkers`.
        """
        # Optimal force bias
        xbar = xp.zeros((walkers.nwalkers, hamiltonian.nfields)) 
        start_time = time.time()
        self.vbias = construct_force_bias(hamiltonian, walkers)
        xbar = -self.sqrt_dt * (1j * self.vbias - self.mf_shift)
        self.timer.tfbias += time.time() - start_time

        # Normally distrubted auxiliary fields.
        xi = xp.random.normal(0.0, 1.0, hamiltonian.nfields * walkers.nwalkers).reshape(
                walkers.nwalkers, hamiltonian.nfields)
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
    
    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=0.):
        start_time = time.time()
        cmf, cfb, xshifted, VHS = self.construct_two_body_propagator(walkers, hamiltonian, trial)
        assert walkers.nwalkers == xshifted.shape[-1]
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3

        start_time = time.time()
        for iw in range(walkers.nwalkers):
            stack = walkers.stack[iw]
            BV = apply_exponential(VHS[iw], self.exp_nmax) # Shape (nbasis, nbasis).
            B = numpy.array([BV.dot(self.BH1[0]), BV.dot(self.BH1[1])])
            B = numpy.array([BV.dot(self.BH1[0]), BV.dot(self.BH1[1])])

            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's Green's function.
            tix = stack.nslice
            G = walkers.greens_function(iw, slice_ix=tix, inplace=False)

            # 2. Compute updated Green's function.
            stack.update_new(B)
            walkers.greens_function(iw, slice_ix=tix, inplace=True)

            # 3. Compute det(G/G')
            # Now apply phaseless approximation
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

