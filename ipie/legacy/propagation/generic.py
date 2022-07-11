import cmath
import math
import sys
import time

import numpy
import scipy.linalg

from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.utils.linalg import exponentiate_matrix, reortho
from ipie.utils.misc import is_cupy


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
    hamiltonian : :class:`pie.hamiltonian.System`
        System object.
    trial : :class:`pie.trial_wavefunctioin.Trial`
        Trial wavefunction object.
    verbose : bool
        If true print out more information during setup.
    """

    def __init__(self, system, hamiltonian, trial, qmc, options={}, verbose=False):
        self.optimised = options.get("optimised", True)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt
        start = time.time()
        if trial.ndets > 1:
            self.optimised = False
            self.mf_shift = self.construct_mean_field_shift_multi_det(
                hamiltonian, trial
            )
        else:
            self.mf_shift = self.construct_mean_field_shift(hamiltonian, trial)

        if verbose:
            print("# Time to mean field shift: {} s".format(time.time() - start))
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )
        # Mean field shifted one-body propagator
        self.construct_one_body_propagator(hamiltonian, qmc.dt)
        # Constant core contribution modified by mean field shift.
        self.mf_core = hamiltonian.ecore + 0.5 * numpy.dot(self.mf_shift, self.mf_shift)
        self.nstblz = qmc.nstblz
        if qmc.batched:
            self.nwalkers = qmc.nwalkers
            self.vbias_batch = numpy.zeros(
                (qmc.nwalkers, hamiltonian.nfields), dtype=numpy.complex128
            )
        else:
            self.vbias = numpy.zeros(hamiltonian.nfields, dtype=numpy.complex128)
        if self.optimised:
            if qmc.batched:
                self.construct_force_bias = None
                self.construct_VHS_batch = self.construct_VHS_batch
            else:
                self.construct_force_bias = self.construct_force_bias_fast
                self.construct_VHS = self.construct_VHS_fast
        else:
            if trial.ndets > 1:
                if qmc.batched:
                    self.construct_force_bias = None
                else:
                    self.construct_force_bias = self.construct_force_bias_multi_det
            else:
                assert qmc.batched == False or qmc.batched == None
                self.construct_force_bias = self.construct_force_bias_slow
            self.construct_VHS = self.construct_VHS_slow
        self.ebound = (2.0 / self.dt) ** 0.5
        self.mean_local_energy = 0
        if verbose:
            print("# Finished setting up propagation.GenericContinuous.")

    def construct_mean_field_shift(self, hamiltonian, trial):
        """Compute mean field shift.

        .. math::

            \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

        """
        # hamiltonian.chol_vecs [X, M^2]
        if hamiltonian.sparse:
            mf_shift = 1j * hamiltonian.chol_vecs * trial.G[0].ravel()
            mf_shift += 1j * hamiltonian.chol_vecs * trial.G[1].ravel()
        else:
            Gcharge = (trial.G[0] + trial.G[1]).ravel()
            if numpy.isrealobj(hamiltonian.chol_vecs):
                tmp_real = numpy.dot(hamiltonian.chol_vecs.T, Gcharge.real)
                tmp_imag = numpy.dot(hamiltonian.chol_vecs.T, Gcharge.imag)
                mf_shift = 1.0j * tmp_real - tmp_imag
            else:
                mf_shift = 1j * numpy.dot(
                    hamiltonian.chol_vecs.T, (trial.G[0] + trial.G[1]).ravel()
                )
        return mf_shift

    def construct_mean_field_shift_multi_det(self, hamiltonian, trial):
        if trial.G is not None:
            mf_shift = self.construct_mean_field_shift(hamiltonian, trial)
        else:
            nb = hamiltonian.nbasis
            mf_shift = [
                trial.contract_one_body(Vpq.reshape(nb, nb))
                for Vpq in hamiltonian.chol_vecs.T
            ]
            mf_shift = 1j * numpy.array(mf_shift)
        return mf_shift

    def construct_one_body_propagator(self, hamiltonian, dt):
        """Construct mean-field shifted one-body propagator.

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
        # shift = 1j*hamiltonian.chol_vecs.dot(self.mf_shift).reshape(nb,nb)
        shift = 1j * numpy.einsum(
            "mx,x->m", hamiltonian.chol_vecs, self.mf_shift
        ).reshape(nb, nb)
        H1 = hamiltonian.h1e_mod - numpy.array([shift, shift])
        self.BH1 = numpy.array(
            [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
        )

    def construct_force_bias_slow(self, hamiltonian, walker, trial):
        """Compute optimal force bias.

        Uses explicit expression.

        Parameters
        ----------
        G: :class:`numpy.ndarray`
            Walker's Green's function.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        # vbias = numpy.einsum('lpq,pq->l', hamiltonian.chol_vecs, walker.G[0])
        # vbias += numpy.einsum('lpq,pq->l', hamiltonian.chol_vecs, walker.G[1])
        vbias = numpy.dot(hamiltonian.chol_vecs.T, walker.G[0].ravel())
        vbias += numpy.dot(hamiltonian.chol_vecs.T, walker.G[1].ravel())
        return -self.sqrt_dt * (1j * vbias - self.mf_shift)

    def construct_force_bias_fast(self, hamiltonian, walker, trial):
        """Compute optimal force bias.

        Uses rotated Green's function.

        Parameters
        ----------
        Ghalf : :class:`numpy.ndarray`
            Half-rotated walker's Green's function.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Force bias.
        """
        G = walker.Ghalf
        if hamiltonian.sparse:
            self.vbias = trial.rot_chol(spin=0) * G[0].ravel()
            self.vbias += trial.rot_chol(spin=1) * G[1].ravel()
        else:
            self.vbias = numpy.dot(trial.rot_chol(spin=0), G[0].ravel())
            self.vbias += numpy.dot(trial.rot_chol(spin=1), G[1].ravel())
        return -self.sqrt_dt * (1j * self.vbias - self.mf_shift)

    def construct_force_bias_multi_det(self, hamiltonian, walker, trial):
        # if (trial.G != None):
        vbias = numpy.array(
            [walker.contract_one_body(Vpq, trial) for Vpq in hamiltonian.chol_vecs.T]
        )
        return -self.sqrt_dt * (1j * vbias - self.mf_shift)
        # else:
        #     return self.construct_force_bias_slow(hamiltonian, walker, trial)

    def construct_VHS_slow(self, hamiltonian, shifted):
        # VHS_{ik} = \sum_{n} v_{(ik),n} (x-xbar)_n
        nb = hamiltonian.nbasis
        return self.isqrt_dt * numpy.einsum(
            "mx,x->m", hamiltonian.chol_vecs, shifted
        ).reshape(nb, nb)

    def construct_VHS_fast(self, hamiltonian, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        if numpy.isrealobj(hamiltonian.chol_vecs):
            VHS = numpy.einsum(
                "mx,x->m", hamiltonian.chol_vecs, xshifted.real
            ) + 1.0j * numpy.einsum("mx,x->m", hamiltonian.chol_vecs, xshifted.imag)
        else:
            # VHS = hamiltonian.chol_vecs.dot(xshifted)
            VHS = numpy.einsum("mx,x->m", hamiltonian.chol_vecs, xshifted)
        VHS = VHS.reshape(hamiltonian.nbasis, hamiltonian.nbasis)
        return self.isqrt_dt * VHS

    def construct_VHS_batch(self, hamiltonian, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        if is_cupy(
            hamiltonian.chol_vecs
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            isrealobj = cupy.isrealobj
        else:
            isrealobj = numpy.isrealobj

        if isrealobj(hamiltonian.chol_vecs):
            VHS = hamiltonian.chol_vecs.dot(
                xshifted.real
            ) + 1.0j * hamiltonian.chol_vecs.dot(xshifted.imag)
        else:
            VHS = hamiltonian.chol_vecs.dot(xshifted)
        # (nb, nb, nw) -> (nw, nb, nb)
        VHS = VHS.T.reshape(
            self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis
        ).copy()

        if is_cupy(
            hamiltonian.chol_vecs
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            cupy.cuda.stream.get_current_stream().synchronize()

        return self.isqrt_dt * VHS


def construct_propagator_matrix_generic(hamiltonian, BT2, config, dt, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with generic hamiltonian object.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    config : numpy array
        Auxiliary field configuration.
    conjt : bool
        If true return Hermitian conjugate of matrix.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full propagator matrix.
    """
    nbsf = hamiltonian.nbasis
    VHS = 1j * dt**0.5 * hamiltonian.chol_vecs.dot(config).reshape(nbsf, nbsf)
    EXP_VHS = exponentiate_matrix(VHS)
    Bup = BT2[0].dot(EXP_VHS).dot(BT2[0])
    Bdown = BT2[1].dot(EXP_VHS).dot(BT2[1])

    if conjt:
        return [Bup.conj().T, Bdown.conj().T]
    else:
        return [Bup, Bdown]


# def back_propagate(system, psi, trial, nstblz, BT2, dt):
# r"""Perform back propagation for RHF/UHF style wavefunction.

# For use with generic system hamiltonian.

# Parameters
# ---------
# system : system object in general.
# Container for model input options.
# psi : :class:`pie.walkers.Walkers` object
# CPMC wavefunction.
# trial : :class:`pie.trial_wavefunction.X' object
# Trial wavefunction class.
# nstblz : int
# Number of steps between GS orthogonalisation.
# BT2 : :class:`numpy.ndarray`
# One body propagator.
# dt : float
# Timestep.

# Returns
# -------
# psi_bp : list of :class:`pie.walker.Walker` objects
# Back propagated list of walkers.
# """
# psi_bp = [SingleDetWalker({}, system, trial, index=w) for w in range(len(psi))]
# nup = system.nup
# for (iw, w) in enumerate(psi):
# # propagators should be applied in reverse order
# for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
# # could make this system specific to reduce need for multiple
# # routines.
# B = construct_propagator_matrix_generic(system, BT2, c, dt, True)
# psi_bp[iw].phi[:,:nup] = B[0].dot(psi_bp[iw].phi[:,:nup])
# psi_bp[iw].phi[:,nup:] = B[1].dot(psi_bp[iw].phi[:,nup:])
# if i != 0 and i % nstblz == 0:
# psi_bp[iw].reortho(trial)
# return psi_bp


def back_propagate_generic(
    phi, configs, system, hamiltonian, nstblz, BT2, dt, store=False
):
    r"""Perform back propagation for RHF/UHF style wavefunction.

    For use with generic system hamiltonian.

    Parameters
    ---------
    system : system object in general.
        Container for model input options.
    psi : :class:`pie.walkers.Walkers` object
        CPMC wavefunction.
    trial : :class:`pie.trial_wavefunction.X' object
        Trial wavefunction class.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    dt : float
        Timestep.

    Returns
    -------
    psi_bp : list of :class:`pie.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs.get_block()[0][::-1]):
        B = construct_propagator_matrix_generic(hamiltonian, BT2, c, dt, False)
        phi[:, :nup] = numpy.dot(B[0].conj().T, phi[:, :nup])
        phi[:, nup:] = numpy.dot(B[1].conj().T, phi[:, nup:])
        if i != 0 and i % nstblz == 0:
            (phi[:, :nup], R) = reortho(phi[:, :nup])
            (phi[:, nup:], R) = reortho(phi[:, nup:])
        if store:
            psi_store.append(phi.copy())

    return psi_store


def back_propagate_generic_bmat(system, psi, trial, nstblz):
    r"""Perform back propagation for RHF/UHF style wavefunction."""
    psi_bp = [
        SingleDetWalker({}, system, hamiltonian, trial, index=w)
        for w in range(len(psi))
    ]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, B) in enumerate(w.stack.stack[::-1]):
            # could make this system specific to reduce need for multiple
            # routines.
            psi_bp[iw].phi[:, :nup] = numpy.dot(B[0].conj().T, psi_bp[iw].phi[:, :nup])
            psi_bp[iw].phi[:, nup:] = numpy.dot(B[1].conj().T, psi_bp[iw].phi[:, nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp
