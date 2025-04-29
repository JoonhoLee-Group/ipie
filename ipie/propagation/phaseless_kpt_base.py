import time
import numpy
import scipy.linalg
from abc import abstractmethod
from ipie.propagation.continuous_base import ContinuousBase
from ipie.propagation.operations import propagate_one_body, propagate_one_body_kpt
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, cast_to_device
import h5py

import plum
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from cuquantum import cutensornet, NetworkOptions, contract
from typing import Union

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from ipie.utils.mpi import make_splits_displacements


@plum.dispatch
def construct_one_body_propagator(
    hamiltonian: KptComplexChol, mf_shift: xp.ndarray, dt: float
):
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
    
    diagchol = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    igamma = hamiltonian.igamma
    for ik in range(hamiltonian.nk):
        diagchol[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :]
            
    shift = xp.einsum("xkpr, x -> kpr", diagchol, mf_shift)
    H1 = hamiltonian.h1e_mod + xp.array([shift, shift])
    if hasattr(H1, "get"):
        H1_numpy = H1.get()
    else:
        H1_numpy = H1

    full_h1 = numpy.zeros((2, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)
    for ik in range(hamiltonian.nk):
        full_h1[0, ik, :, ik, :] = H1_numpy[0, ik]
        full_h1[1, ik, :, ik, :] = H1_numpy[1, ik]
    full_h1_mat = full_h1.reshape(2, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
    expH1 = xp.array(
        [scipy.linalg.expm(-0.5 * dt * full_h1_mat[0]), scipy.linalg.expm(-0.5 * dt * full_h1_mat[1])]
    )
    return expH1

@plum.dispatch
def construct_one_body_propagator(
    hamiltonian: Union[KptComplexCholSymm, KptComplexCholChunked], mf_shift: xp.ndarray, dt: float
):
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
    igamma = hamiltonian.igamma
    
    if hamiltonian.chunked:
        start_n = hamiltonian.chunk_displacements[hamiltonian.handler.srank]
        end_n = hamiltonian.chunk_displacements[hamiltonian.handler.srank + 1]
        diagcholchunk = numpy.zeros((end_n - start_n, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
        diagcholdaggerchunk = numpy.zeros((end_n - start_n, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)

        for ik in range(hamiltonian.nk):
            diagcholchunk[:, ik, :, :] = hamiltonian.chol_chunk[:, ik, :, igamma, :]
            diagcholdaggerchunk[:, ik, :, :] = hamiltonian.chol_chunk[:, ik, :, igamma, :].transpose(0, 2, 1).conj()

        mf_shiftL = mf_shift[:hamiltonian.nchol]
        mf_shiftLdag = mf_shift[hamiltonian.nchol:]
        shift = .5 * (xp.einsum("xkpr, x -> kpr", diagcholchunk, mf_shiftLdag[start_n:end_n]) + xp.einsum("xkpr, x -> kpr", diagcholdaggerchunk, mf_shiftL[start_n:end_n]))
        if MPI is None:
            raise ImportError("mpi4py is not installed.")
        else:
            shift = hamiltonian.handler.scomm.allreduce(shift, op=MPI.SUM)
    else:
        diagchol = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
        diagcholdagger = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
        
        for ik in range(hamiltonian.nk):
            diagchol[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :]
            diagcholdagger[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :].transpose(0, 2, 1).conj()

        mf_shiftL = mf_shift[:hamiltonian.nchol]
        mf_shiftLdag = mf_shift[hamiltonian.nchol:]
        shift = .5 * (xp.einsum("xkpr, x -> kpr", diagchol, mf_shiftLdag) + xp.einsum("xkpr, x -> kpr", diagcholdagger, mf_shiftL))
    H1 = hamiltonian.h1e_mod + xp.array([shift, shift])
    if hasattr(H1, "get"):
        H1_numpy = H1.get()
    else:
        H1_numpy = H1

    expH1_0 = numpy.zeros((hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    expH1_1 = numpy.zeros((hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    for ik in range(hamiltonian.nk):
        expH1_0[ik] = scipy.linalg.expm(-0.5 * dt * H1_numpy[0, ik])
        expH1_1[ik] = scipy.linalg.expm(-0.5 * dt * H1_numpy[1, ik])
    expH1 = xp.array([expH1_0, expH1_1])
    return expH1

@plum.dispatch
def construct_mean_field_shift(hamiltonian: KptComplexChol, trial: KptSingleDet):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    Remark: Here the convention is a little different because mf_shift without the 1j is more convenient.

    """
    # trial G [nk, nbsf, nbsf]
    diagchol = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    igamma = hamiltonian.igamma
    for ik in range(hamiltonian.nk):
        diagchol[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :]
    diagchol = diagchol.reshape(hamiltonian.nchol, hamiltonian.nk * hamiltonian.nbasis * hamiltonian.nbasis)
    Gcharge = (trial.G[0] + trial.G[1]).ravel()
    mf_shift = numpy.dot(diagchol, Gcharge)
    return xp.array(mf_shift)

@plum.dispatch
def construct_mean_field_shift(hamiltonian: KptComplexCholSymm, trial: KptSingleDet):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    Remark: Here the convention is a little different because mf_shift without the 1j is more convenient.

    """
    # trial G [nk, nbsf, nbsf]
    diagchol = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    diagcholdagger = numpy.zeros((hamiltonian.nchol, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    igamma = hamiltonian.igamma
    for ik in range(hamiltonian.nk):
        diagchol[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :]
        diagcholdagger[:, ik, :, :] = hamiltonian.chol[:, ik, :, igamma, :].transpose(0, 2, 1).conj()
    diagchol = diagchol.reshape(hamiltonian.nchol, hamiltonian.nk * hamiltonian.nbasis * hamiltonian.nbasis)
    diagcholdagger = diagcholdagger.reshape(hamiltonian.nchol, hamiltonian.nk * hamiltonian.nbasis * hamiltonian.nbasis)
    Gcharge = (trial.G[0] + trial.G[1]).ravel()
    mf_shiftL = numpy.dot(diagchol, Gcharge)
    mf_shiftLconj = numpy.dot(diagcholdagger, Gcharge)

    mf_shift = xp.concatenate((mf_shiftL, mf_shiftLconj))
    return xp.array(mf_shift)

@plum.dispatch
def construct_mean_field_shift(hamiltonian: KptComplexCholChunked, trial: KptSingleDet):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    Remark: Here the convention is a little different because mf_shift without the 1j is more convenient.

    """
    igamma = hamiltonian.igamma
    ncholchunk = hamiltonian.chol_chunk.shape[0]
    diagcholchunk = numpy.zeros((ncholchunk, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    diagcholdaggerchunk = numpy.zeros((ncholchunk, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)

    for ik in range(hamiltonian.nk):
        diagcholchunk[:, ik, :, :] = hamiltonian.chol_chunk[:, ik, :, igamma, :]
        diagcholdaggerchunk[:, ik, :, :] = hamiltonian.chol_chunk[:, ik, :, igamma, :].transpose(0, 2, 1).conj()
    
    diagcholchunk = diagcholchunk.reshape(-1, hamiltonian.nk * hamiltonian.nbasis * hamiltonian.nbasis)
    diagcholdaggerchunk = diagcholdaggerchunk.reshape(-1, hamiltonian.nk * hamiltonian.nbasis * hamiltonian.nbasis)
    Gcharge = (trial.G[0] + trial.G[1]).ravel()
    mf_shiftL = numpy.dot(diagcholchunk, Gcharge)
    mf_shiftLconj = numpy.dot(diagcholdaggerchunk, Gcharge)

    split_sizes, displacements = make_splits_displacements(hamiltonian.nchol, trial.handler.ssize)
    split_sizes_np = numpy.array(split_sizes, dtype=int)
    displacements_np = numpy.array(displacements, dtype=int)

    recvbuf = numpy.zeros(hamiltonian.nchol, dtype=numpy.complex128)
    recvbuf_conj = numpy.zeros(hamiltonian.nchol, dtype=numpy.complex128)
    if MPI is None:
        raise ImportError("mpi4py is not installed.")
    else:
        trial.handler.scomm.Allgatherv(mf_shiftL, [recvbuf, split_sizes_np, displacements_np, MPI.DOUBLE_COMPLEX])
        trial.handler.scomm.Allgatherv(mf_shiftLconj, [recvbuf_conj, split_sizes_np, displacements_np, MPI.DOUBLE_COMPLEX])

    trial.handler.scomm.Bcast(recvbuf, root=0)
    trial.handler.scomm.Bcast(recvbuf_conj, root=0)

    mf_shift = numpy.concatenate((recvbuf, recvbuf_conj))
    return xp.array(mf_shift)

@plum.dispatch
def construct_mean_field_shift(hamiltonian: KptISDF, trial: KptSingleDet):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    Remark: Here the convention is a little different because mf_shift without the 1j is more convenient.

    """
    # trial G [nk, nbsf, nbsf]
    igamma = hamiltonian.igamma
    diagcholM = hamiltonian.cholM[igamma] # [nisdf, naux]
    cgto = hamiltonian.cgto
    Gcharge = (trial.G[0] + trial.G[1])
    # convert to complex128
    Gcharge = Gcharge.astype(numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    mf_shift = contract("kPp, kPr, Pg, kpr -> g", cgto.conj(), cgto, diagcholM, Gcharge, options=network_opts)
    cutensornet.destroy(handle)
    return xp.array(mf_shift)

@plum.dispatch
def construct_one_body_propagator(
    hamiltonian: KptISDF, mf_shift: xp.ndarray, dt: float
):
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
    
    cholpcholconj = hamiltonian.cholM + hamiltonian.cholM.conj()
    igamma = hamiltonian.igamma
    cgto = hamiltonian.cgto
    # to cupy array
    cholpcholconj = xp.array(cholpcholconj)
    cgto = xp.array(cgto)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    shift = .5 * contract("kPp, kPq, Pg, g -> kpq", cgto.conj(), cgto, cholpcholconj[igamma], mf_shift, options=network_opts)
    cutensornet.destroy(handle)
    H1 = hamiltonian.h1e_mod + xp.array([shift, shift])
    if hasattr(H1, "get"):
        H1_numpy = H1.get()
    else:
        H1_numpy = H1

    expH1_0 = numpy.zeros((hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    expH1_1 = numpy.zeros((hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128)
    for ik in range(hamiltonian.nk):
        expH1_0[ik] = scipy.linalg.expm(-0.5 * dt * H1_numpy[0, ik])
        expH1_1[ik] = scipy.linalg.expm(-0.5 * dt * H1_numpy[1, ik])
    expH1 = xp.array([expH1_0, expH1_1])
    return expH1


@plum.dispatch
def construct_mf_mod_xbar(hamiltonian: KptComplexChol, mf_shift: xp.ndarray):
    """
    Modify xbar using mean field shift for KptComplexChol Hamiltonian.
    """
    mf_xbarp = 1j * mf_shift
    mf_xbarm = xp.zeros_like(mf_xbarp)
    return xp.array([mf_xbarp, mf_xbarm])

@plum.dispatch
def construct_mf_mod_xbar(hamiltonian: Union[KptComplexCholSymm, KptComplexCholChunked], mf_shift: xp.ndarray):
    """
    Modify xbar using mean field shift for KptComplexCholSymm Hamiltonian.
    """
    mf_L = mf_shift[:hamiltonian.nchol]
    mf_Lconj = mf_shift[hamiltonian.nchol:]
    mf_xbarp = .5j * (mf_L + mf_Lconj)
    mf_xbarm = .5 * (mf_L - mf_Lconj)
    return xp.array([mf_xbarp, mf_xbarm])

@plum.dispatch
def construct_mf_mod_xbar(hamiltonian: KptISDF, mf_shift: xp.ndarray):
    """
    Modify xbar using mean field shift for KptISDF Hamiltonian.
    """
    mf_xbarp = 1j * mf_shift
    mf_xbarm = xp.zeros_like(mf_xbarp)
    return xp.array([mf_xbarp, mf_xbarm])

class PhaselessKptBase(ContinuousBase):
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


    def propagate_walkers_one_body(self, walkers, hamiltonian):
        start_time = time.time()
        phia_reshaped = walkers.phia.reshape(walkers.nwalkers, hamiltonian.nk, hamiltonian.nbasis, -1)
        phia = propagate_one_body_kpt(phia_reshaped, self.expH1[0])
        walkers.phia = phia.reshape(walkers.nwalkers, hamiltonian.nk * hamiltonian.nbasis, -1)
        if walkers.ndown > 0 and not walkers.rhf:
            phib_reshaped = walkers.phib.reshape(walkers.nwalkers, hamiltonian.nk, hamiltonian.nbasis, -1)
            phib = propagate_one_body_kpt(phib_reshaped, self.expH1[1])
            walkers.phib = phib.reshape(walkers.nwalkers, hamiltonian.nk * hamiltonian.nbasis, -1)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        # optimal force bias
        xbar = xp.zeros((2, walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)

        start_time = time.time()
        self.vbias_plus, self.vbias_minus = trial.calc_force_bias(hamiltonian, walkers, walkers.mpi_handler)

        xbar_plus = numpy.zeros_like(self.vbias)
        igamma = hamiltonian.igamma
        mf_xbarp, mf_xbarm = construct_mf_mod_xbar(hamiltonian, self.mf_shift)

        xbar_plus = -self.sqrt_dt * self.vbias_plus
        xbar_plus[:, :, igamma] = -self.sqrt_dt * (self.vbias_plus[:, :, igamma] - mf_xbarp[numpy.newaxis, :]) 
        xbar_minus = -self.sqrt_dt * self.vbias_minus
        xbar_minus[:, :, igamma] = -self.sqrt_dt * (self.vbias_minus[:, :, igamma] - mf_xbarm[numpy.newaxis, :])
        xbar[0] = xbar_plus
        xbar[1] = xbar_minus
        synchronize()
        self.timer.tfbias += time.time() - start_time

        # force bias bounding
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)

        # Normally distributed auxiliary fields.

        xi = xp.random.normal(0.0, 1.0, 2 * hamiltonian.nchol * hamiltonian.unique_nk * walkers.nwalkers).reshape(2, walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk)

        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        xshifted_plus_q0 = xshifted[0, :, :, hamiltonian.igamma]
        xshifted_minus_q0 = xshifted[1, :, :, hamiltonian.igamma]
        cmf = - self.sqrt_dt * xp.einsum("wx,x->w", xshifted_plus_q0, mf_xbarp) - self.sqrt_dt * xp.einsum("wx,x->w", xshifted_minus_q0, mf_xbarm)

        cfb = xp.einsum("swxq,swxq->w", xi, xbar) - 0.5 * xp.einsum("swxq,swxq->w", xbar, xbar)

        self.apply_VHS(walkers, hamiltonian, xshifted)

        # xp._default_memory_pool.free_all_blocks()
        return (cmf, cfb)

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        start_time = time.time()
        _, sgn_ovlp, log_ovlp = trial.calc_greens_function(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Slater matrix
        # 2.a Apply one-body
        self.propagate_walkers_one_body(walkers, hamiltonian)

        # 2.b Apply two-body
        (cmf, cfb) = self.propagate_walkers_two_body(walkers, hamiltonian, trial)

        # 2.c Apply one-body
        self.propagate_walkers_one_body(walkers, hamiltonian)

        # Now apply phaseless approximation
        start_time = time.time()
        _, sgn_ovlpnew, log_ovlpnew = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, sgn_ovlp, log_ovlp, sgn_ovlpnew, log_ovlpnew, cfb, cmf, eshift)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def update_weight(self, walkers, sgn_ovlp, log_ovlp, sgn_ovlpnew, log_ovlpnew, cfb, cmf, eshift):
        ovlp_ratio = sgn_ovlpnew / sgn_ovlp * xp.exp(log_ovlpnew - log_ovlp)
        hybrid_energy = -(xp.log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self.apply_bound_hybrid(hybrid_energy, eshift)
        importance_function = xp.exp(
            -self.dt * (0.5 * (hybrid_energy + walkers.hybrid_energy) - eshift)
        )
        magn = xp.abs(importance_function)
        walkers.hybrid_energy = xp.where(
            xp.isnan(hybrid_energy),       
            walkers.hybrid_energy,        
            hybrid_energy
        )


        dtheta = (-self.dt * hybrid_energy - cfb).imag
        cosine_fac = xp.cos(dtheta)

        xp.clip(
            cosine_fac, a_min=0.0, a_max=None, out=cosine_fac
        )  # in-place clipping (cosine projection)
        walkers.weight = walkers.weight * magn * cosine_fac
        walkers.ovlp = sgn_ovlpnew * xp.exp(log_ovlpnew - walkers.log_shift)
        walkers.sgn_ovlp = sgn_ovlpnew
        walkers.log_ovlp = log_ovlpnew

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

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)
