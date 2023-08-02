import math
import time

import numpy

from ipie.utils.pack_numba import unpack_VHS_batch

try:
    from ipie.utils.pack_numba_gpu import unpack_VHS_batch_gpu
except:
    pass

import plum

from ipie.config import config
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers


class PhaselessGeneric(PhaselessBase):
    """A class for performing phaseless propagation with real, generic, hamiltonian."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        start_time = time.time()
        assert walkers.nwalkers == xshifted.shape[-1]
        VHS = self.construct_VHS(hamiltonian, xshifted)
        synchronize()
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3

        start_time = time.time()
        if config.get_option("use_gpu"):
            walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
        else:
            for iw in range(walkers.nwalkers):
                # 2.b Apply two-body
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        print("JOONHO here abstract function for construct VHS")
        "abstract function for construct VHS"

    # Any class inherited from PhaselessGeneric should override this method.
    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericRealChol, xshifted: xp.ndarray) -> xp.ndarray:
        nwalkers = xshifted.shape[-1]

        VHS_packed = hamiltonian.chol_packed.dot(
            xshifted.real
        ) + 1.0j * hamiltonian.chol_packed.dot(xshifted.imag)

        # (nb, nb, nw) -> (nw, nb, nb)
        VHS_packed = (
            self.isqrt_dt * VHS_packed.T.reshape(nwalkers, hamiltonian.chol_packed.shape[0]).copy()
        )

        VHS = xp.zeros(
            (nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=VHS_packed.dtype,
        )
        if config.get_option("use_gpu"):
            threadsperblock = 512
            nbsf = hamiltonian.nbasis
            nut = round(nbsf * (nbsf + 1) / 2)
            blockspergrid = math.ceil(nwalkers * nut / threadsperblock)
            unpack_VHS_batch_gpu[blockspergrid, threadsperblock](
                hamiltonian.sym_idx_i, hamiltonian.sym_idx_j, VHS_packed, VHS
            )
        else:
            unpack_VHS_batch(hamiltonian.sym_idx[0], hamiltonian.sym_idx[1], VHS_packed, VHS)
        return VHS

    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericComplexChol, xshifted: xp.ndarray) -> xp.ndarray:
        nwalkers = xshifted.shape[-1]

        nchol = hamiltonian.nchol

        VHS = self.isqrt_dt * (
            hamiltonian.A.dot(xshifted[:nchol]) + hamiltonian.B.dot(xshifted[nchol:])
        )
        VHS = VHS.T.copy()
        VHS = VHS.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)

        return VHS


class PhaselessGenericChunked(PhaselessGeneric):
    """A class for performing phaseless propagation with real, generic, hamiltonian."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, exp_nmax=exp_nmax, verbose=verbose)

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        super().build(hamiltonian, trial, walkers, mpi_handler, verbose)
        self.mpi_handler = mpi_handler

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        "abstract function for construct VHS"

    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericRealChol, xshifted: xp.ndarray) -> xp.ndarray:
        assert hamiltonian.chunked
        assert xp.isrealobj(hamiltonian.chol)

        nwalkers = xshifted.shape[-1]

        # if hamiltonian.mixed_precision:  # cast it to float
        #     xshifted = xshifted.astype(numpy.complex64)

        #       xshifted is unique for each processor!
        xshifted_send = xshifted.copy()
        xshifted_recv = xp.zeros_like(xshifted)

        idxs = hamiltonian.chol_idxs_chunk
        chol_packed_chunk = hamiltonian.chol_packed_chunk

        VHS_send = chol_packed_chunk.dot(xshifted[idxs, :].real) + 1.0j * chol_packed_chunk.dot(
            xshifted[idxs, :].imag
        )
        VHS_recv = xp.zeros_like(VHS_send)

        srank = self.mpi_handler.scomm.rank
        sender = numpy.where(self.mpi_handler.receivers == srank)[0]

        for _ in range(self.mpi_handler.ssize - 1):
            synchronize()
            self.mpi_handler.scomm.Isend(
                xshifted_send, dest=self.mpi_handler.receivers[srank], tag=1
            )
            self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=2)

            req1 = self.mpi_handler.scomm.Irecv(xshifted_recv, source=sender, tag=1)
            req2 = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=2)
            req1.wait()
            req2.wait()

            self.mpi_handler.scomm.barrier()

            # prepare sending
            VHS_send = (
                VHS_recv
                + chol_packed_chunk.dot(xshifted_recv[idxs, :].real)
                + 1.0j * chol_packed_chunk.dot(xshifted_recv[idxs, :].imag)
            )
            xshifted_send = xshifted_recv.copy()

        synchronize()
        self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=1)
        req = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=1)
        req.wait()
        self.mpi_handler.scomm.barrier()

        VHS_recv = self.isqrt_dt * VHS_recv.T.reshape(nwalkers, chol_packed_chunk.shape[0]).copy()
        VHS = xp.zeros(
            (nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=VHS_recv.dtype,
        )
        # This should be abstracted by kernel import
        if config.get_option("use_gpu"):
            threadsperblock = 512
            nut = len(hamiltonian.sym_idx_i)
            blockspergrid = math.ceil(nwalkers * nut / threadsperblock)
            unpack_VHS_batch_gpu[blockspergrid, threadsperblock](
                hamiltonian.sym_idx_i, hamiltonian.sym_idx_j, VHS_recv, VHS
            )
        else:
            unpack_VHS_batch(hamiltonian.sym_idx[0], hamiltonian.sym_idx[1], VHS_recv, VHS)
        synchronize()
        return VHS


Phaseless = {"generic": PhaselessGeneric, "chunked": PhaselessGenericChunked}
