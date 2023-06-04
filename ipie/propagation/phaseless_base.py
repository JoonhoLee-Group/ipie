import time
import numpy
import scipy.linalg
from abc import abstractmethod
from ipie.propagation.continuous_base import ContinuousBase
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

import plum
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol


@plum.dispatch
def construct_one_body_propagator(hamiltonian: GenericRealChol, mf_shift: xp.ndarray, dt: float):
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
    shift = 1j * numpy.einsum("mx,x->m", hamiltonian.chol, mf_shift).reshape(nb, nb)
    H1 = hamiltonian.h1e_mod - numpy.array([shift, shift])
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )
    return expH1


@plum.dispatch
def construct_one_body_propagator(hamiltonian: GenericComplexChol, mf_shift: xp.ndarray, dt: float):
    nb = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    shift = xp.zeros((nb, nb), dtype=hamiltonian.chol.dtype)
    shift = 1j * numpy.einsum("mx,x->m", hamiltonian.A, mf_shift[:nchol]).reshape(nb, nb)

    shift += 1j * numpy.einsum("mx,x->m", hamiltonian.B, mf_shift[nchol:]).reshape(nb, nb)

    H1 = hamiltonian.h1e_mod - numpy.array([shift, shift])
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )
    return expH1


@plum.dispatch
def construct_mean_field_shift(hamiltonian: GenericRealChol, trial: TrialWavefunctionBase):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    """
    # hamiltonian.chol [X, M^2]
    Gcharge = (trial.G[0] + trial.G[1]).ravel()
    tmp_real = numpy.dot(hamiltonian.chol.T, Gcharge.real)
    tmp_imag = numpy.dot(hamiltonian.chol.T, Gcharge.imag)
    mf_shift = 1.0j * tmp_real - tmp_imag
    return mf_shift


@plum.dispatch
def construct_mean_field_shift(hamiltonian: GenericComplexChol, trial: TrialWavefunctionBase):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    """
    # hamiltonian.chol [X, M^2]
    Gcharge = (trial.G[0] + trial.G[1]).ravel()

    nchol = hamiltonian.nchol
    nfields = hamiltonian.nfields

    mf_shift = numpy.zeros(nfields, dtype=hamiltonian.chol.dtype)
    mf_shift[:nchol] = 1j * numpy.dot(hamiltonian.A.T, Gcharge.ravel())
    mf_shift[nchol:] = 1j * numpy.dot(hamiltonian.B.T, Gcharge.ravel())
    return mf_shift


class PhaselessBase(ContinuousBase):
    """A base class for generic continuous HS transform AFQMC propagators."""

    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.sqrt_dt = self.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt

        self.nfb_trig = 0  # number of force bias triggered
        self.nhe_trig = 0  # number of hybrid enerby bound triggered
        self.ebound = (2.0 / self.dt) ** 0.5  # energy bound range
        self.fbbound = 1.0
        self.mpi_handler = None

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
        # construct one-body propagator
        self.expH1 = construct_one_body_propagator(hamiltonian, self.mf_shift, self.dt)

        # # Allocate force bias (we don't need to do this here - it will be allocated when it is needed)
        self.vbias = None
        # self.vbias = numpy.zeros((walkers.nwalkers, hamiltonian.nfields),
        #                         dtype=numpy.complex128)

    def propagate_walkers_one_body(self, walkers):
        start_time = time.time()
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        if walkers.ndown > 0 and not walkers.rhf:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        # optimal force bias
        xbar = xp.zeros((walkers.nwalkers, hamiltonian.nfields))

        start_time = time.time()
        self.vbias = trial.calc_force_bias(hamiltonian, walkers, walkers.mpi_handler)
        xbar = -self.sqrt_dt * (1j * self.vbias - self.mf_shift)
        synchronize()
        self.timer.tfbias += time.time() - start_time

        # force bias bounding
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)

        # Normally distrubted auxiliary fields.
        xi = xp.random.normal(0.0, 1.0, hamiltonian.nfields * walkers.nwalkers).reshape(
            walkers.nwalkers, hamiltonian.nfields
        )
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted, self.mf_shift)
        # Constant factor arising from shifting the propability distribution.
        cfb = xp.einsum("wx,wx->w", xi, xbar) - 0.5 * xp.einsum("wx,wx->w", xbar, xbar)

        xshifted = xshifted.T.copy()

        self.apply_VHS(walkers, hamiltonian, xshifted)

        return (cmf, cfb)

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Slater matrix
        # 2.a Apply one-body
        self.propagate_walkers_one_body(walkers)

        # 2.b Apply two-body
        (cmf, cfb) = self.propagate_walkers_two_body(walkers, hamiltonian, trial)

        # 2.c Apply one-body
        self.propagate_walkers_one_body(walkers)

        # Now apply phaseless approximation
        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new, cfb, cmf, eshift)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def update_weight(self, walkers, ovlp, ovlp_new, cfb, cmf, eshift):
        ovlp_ratio = ovlp_new / ovlp
        hybrid_energy = -(xp.log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self.apply_bound_hybrid(hybrid_energy, eshift)
        importance_function = xp.exp(
            -self.dt * (0.5 * (hybrid_energy + walkers.hybrid_energy) - eshift)
        )
        # splitting w_alpha = |I(x,\bar{x},|phi_alpha>)| e^{i theta_alpha}
        magn = xp.abs(importance_function)
        walkers.hybrid_energy = hybrid_energy

        dtheta = (-self.dt * hybrid_energy - cfb).imag
        cosine_fac = xp.cos(dtheta)

        xp.clip(
            cosine_fac, a_min=0.0, a_max=None, out=cosine_fac
        )  # in-place clipping (cosine projection)
        walkers.weight = walkers.weight * magn * cosine_fac
        walkers.ovlp = ovlp_new

    def apply_bound_force_bias(self, xbar, max_bound=1.0):
        absxbar = xp.abs(xbar)
        idx_to_rescale = absxbar > max_bound
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.copy()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        xbar = xp.where(idx_to_rescale, xbar_rescaled, xbar)
        self.nfb_trig += xp.sum(idx_to_rescale)
        return xbar

    def apply_bound_hybrid(self, ehyb, eshift):  # shift is a number but ehyb is not
        # For initial steps until first estimator communication eshift will be
        # zero and hybrid energy can be incorrect. So just avoid capping for
        # first block until reasonable estimate of eshift can be computed.
        if abs(eshift) < 1e-10:
            return ehyb
        emax = eshift.real + self.ebound
        emin = eshift.real - self.ebound
        xp.clip(ehyb.real, a_min=emin, a_max=emax, out=ehyb.real)  # in-place clipping
        synchronize()
        return ehyb

    # form and apply VHS to walkers
    @abstractmethod
    def apply_VHS(self, walkers, hamiltonian, xshifted):
        pass
