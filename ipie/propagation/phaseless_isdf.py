import time
from typing import Union

import numpy
import plum

from ipie.config import config
from ipie.hamiltonians.generic_base import GenericBase
from ipie.hamiltonians.chunked_isdf import GenericRealISDFChunked
from ipie.hamiltonians.isdf import GenericComplexISDF, GenericRealISDF
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.cuquantum_backend import (
    NetworkOptions_optional as NetworkOptions,
    contract_optional as contract,
    cutensornet_optional as cutensornet,
)
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.walkers.uhf_walkers import UHFWalkers


def apply_VHS_to_phi_batch(cgto, Lx, phi, handle):
    network_opts = NetworkOptions(handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0])
    outphi = contract(
        "Pw, Pp, Pr, wri -> wpi", Lx, cgto, cgto, phi, handle=handle, options=network_opts
    )
    xp._default_memory_pool.free_all_blocks()
    return outphi


class PhaselessISDF(PhaselessBase):
    """A class for performing phaseless propagation with generic ISDF hamiltonian."""

    def __init__(self, time_step, ebound_const=2.0, fbbound=1.0, exp_nmax=6, verbose=False):
        super().__init__(time_step, ebound_const=ebound_const, fbbound=fbbound, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(
        self,
        walkers: Union[UHFWalkers, GHFWalkers],
        hamiltonian: Union[GenericRealISDF, GenericComplexISDF],
        xshifted: xp.ndarray,
    ):
        if config.get_option("use_gpu"):
            nbsf = hamiltonian.nbasis
            nwalkers = walkers.nwalkers
            occ_ratio = (
                walkers.nup / walkers.nbasis
                if walkers.rhf
                else (walkers.nup + walkers.ndown) / (walkers.nbasis)
            )
            mem_vhs = 2 * nwalkers * nbsf**2 * 16
            if mem_vhs > 0.35 * xp.cuda.Device().mem_info[0] or occ_ratio < 0.2:
                start_time = time.time()
                assert walkers.nwalkers == xshifted.shape[-1]
                Lx = self.contract_cholM_xshifted(hamiltonian, xshifted)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(VHS.shape) == 3

                start_time = time.time()
                Temp = xp.zeros(walkers.phia.shape, dtype=walkers.phia.dtype)
                xp.copyto(Temp, walkers.phia)
                handle = cutensornet.create()
                for n in range(1, self.exp_nmax + 1):
                    Temp = (
                        apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Temp, handle) / n
                    )  # matmul use much less GPU memory than einsum
                    walkers.phia += Temp
                del Temp
                if walkers.ndown > 0 and not walkers.rhf:
                    Temp = xp.zeros(walkers.phib.shape, dtype=walkers.phib.dtype)
                    xp.copyto(Temp, walkers.phib)
                    handle = cutensornet.create()
                    for n in range(1, self.exp_nmax + 1):
                        Temp = (
                            apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Temp, handle) / n
                        )  # matmul use much less GPU memory than einsum
                        walkers.phib += Temp
                    del Temp
                synchronize()
                self.timer.tgemm += time.time() - start_time
            else:
                start_time = time.time()
                assert walkers.nwalkers == xshifted.shape[-1]
                VHS = self.construct_VHS(hamiltonian, xshifted)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(VHS.shape) == 3

                start_time = time.time()
                walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
                synchronize()
                self.timer.tgemm += time.time() - start_time
        else:
            start_time = time.time()
            assert walkers.nwalkers == xshifted.shape[-1]
            VHS = self.construct_VHS(hamiltonian, xshifted)
            synchronize()
            self.timer.tvhs += time.time() - start_time
            assert len(VHS.shape) == 3

            start_time = time.time()
            for iw in range(walkers.nwalkers):
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
            synchronize()
            self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        """Abstract fallback for ``construct_VHS``.

        Dispatched to when no specialised overload matches the Hamiltonian type.
        """
        print("JOONHO here abstract function for construct VHS")

    @plum.dispatch
    def construct_VHS(
        self, hamiltonian: Union[GenericRealISDF, GenericComplexISDF], xshifted: xp.ndarray
    ) -> xp.ndarray:
        """Construct the Hubbard-Stratonovich operator ``V_HS`` from ISDF factors.

        Parameters
        ----------
        hamiltonian : :class:`~ipie.hamiltonians.isdf.GenericRealISDF` or :class:`~ipie.hamiltonians.isdf.GenericComplexISDF`
            ISDF-factorised Hamiltonian.
        xshifted : :class:`numpy.ndarray`
            Force-bias shifted auxiliary fields, shape ``(nwalkers, nfields)``.

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            One-body operators, one per walker.
        """
        if isinstance(hamiltonian, GenericRealISDF):
            Lx_real = hamiltonian.cholM @ xshifted.real
            Lx_imag = hamiltonian.cholM @ xshifted.imag
            nwalkers = xshifted.shape[-1]
            handle = cutensornet.create()
            network_opts = NetworkOptions(
                handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0]
            )
            VHS_real = contract(
                "Pw, Pp, Pr -> wpr",
                Lx_real,
                hamiltonian.cgto,
                hamiltonian.cgto,
                options=network_opts,
            )
            VHS_imag = contract(
                "Pw, Pp, Pr -> wpr",
                Lx_imag,
                hamiltonian.cgto,
                hamiltonian.cgto,
                options=network_opts,
            )
            VHS = xp.zeros((nwalkers, hamiltonian.nbasis, hamiltonian.nbasis), dtype=xp.complex128)
            VHS.real = VHS_real
            VHS.imag = VHS_imag
            VHS = self.isqrt_dt * VHS
            synchronize()
            xp._default_memory_pool.free_all_blocks()
        elif isinstance(hamiltonian, GenericComplexISDF):
            Lx = hamiltonian.cholM @ xshifted
            handle = cutensornet.create()
            network_opts = NetworkOptions(
                handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0]
            )
            VHS = contract(
                "Pw, Pp, Pr -> wpr", Lx, hamiltonian.cgto, hamiltonian.cgto, options=network_opts
            )
            VHS = self.isqrt_dt * VHS
            synchronize()
            xp._default_memory_pool.free_all_blocks()
        else:
            raise ValueError("Invalid hamiltonian type")
        return VHS

    def contract_cholM_xshifted(
        self, hamiltonian: Union[GenericRealISDF, GenericComplexISDF], xshifted: xp.ndarray
    ) -> xp.ndarray:
        if isinstance(hamiltonian, GenericRealISDF):
            Lx_real = hamiltonian.cholM @ xshifted.real
            Lx_imag = hamiltonian.cholM @ xshifted.imag
            Lx = xp.zeros(Lx_real.shape, dtype=xp.complex128)
            Lx.real = Lx_real
            Lx.imag = Lx_imag
        elif isinstance(hamiltonian, GenericComplexISDF):
            Lx = hamiltonian.cholM @ xshifted
        else:
            raise ValueError("Invalid hamiltonian type")
        return Lx


class PhaselessISDFChunked(PhaselessBase):
    """A class for performing phaseless propagation with generic ISDF hamiltonian."""

    def __init__(self, time_step, ebound_const=2.0, fbbound=1.0, exp_nmax=6, verbose=False):
        super().__init__(time_step, ebound_const=ebound_const, fbbound=fbbound, verbose=verbose)
        self.exp_nmax = exp_nmax

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        super().build(hamiltonian, trial, walkers, mpi_handler, verbose)
        self.mpi_handler = mpi_handler

    @plum.dispatch
    def apply_VHS(
        self,
        walkers: Union[UHFWalkers, GHFWalkers],
        hamiltonian: GenericRealISDFChunked,
        xshifted: xp.ndarray,
    ):
        if config.get_option("use_gpu"):
            nbsf = hamiltonian.nbasis
            nwalkers = walkers.nwalkers
            occ_ratio = (
                walkers.nup / walkers.nbasis
                if walkers.rhf
                else (walkers.nup + walkers.ndown) / (walkers.nbasis)
            )
            mem_vhs = 2 * nwalkers * nbsf**2 * 16
            if mem_vhs > 0.35 * xp.cuda.Device().mem_info[0] or occ_ratio < 0.2:
                start_time = time.time()
                assert walkers.nwalkers == xshifted.shape[-1]
                Lx = self.contract_cholM_xshifted_chunked(hamiltonian, xshifted)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(VHS.shape) == 3

                start_time = time.time()
                Temp = xp.zeros(walkers.phia.shape, dtype=walkers.phia.dtype)
                xp.copyto(Temp, walkers.phia)
                handle = cutensornet.create()
                for n in range(1, self.exp_nmax + 1):
                    Temp = (
                        apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Temp, handle) / n
                    )  # matmul use much less GPU memory than einsum
                    walkers.phia += Temp
                del Temp
                if walkers.ndown > 0 and not walkers.rhf:
                    Temp = xp.zeros(walkers.phib.shape, dtype=walkers.phib.dtype)
                    xp.copyto(Temp, walkers.phib)
                    handle = cutensornet.create()
                    for n in range(1, self.exp_nmax + 1):
                        Temp = (
                            apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Temp, handle) / n
                        )  # matmul use much less GPU memory than einsum
                        walkers.phib += Temp
                    del Temp
                synchronize()
                self.timer.tgemm += time.time() - start_time
            else:
                start_time = time.time()
                assert walkers.nwalkers == xshifted.shape[-1]
                VHS = self.construct_VHS(hamiltonian, xshifted)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(VHS.shape) == 3

                start_time = time.time()
                walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
                synchronize()
                self.timer.tgemm += time.time() - start_time
        else:
            start_time = time.time()
            assert walkers.nwalkers == xshifted.shape[-1]
            VHS = self.construct_VHS(hamiltonian, xshifted)
            synchronize()
            self.timer.tvhs += time.time() - start_time
            assert len(VHS.shape) == 3

            start_time = time.time()
            for iw in range(walkers.nwalkers):
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
            synchronize()
            self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        """Abstract fallback for ``construct_VHS``.

        Dispatched to when no specialised overload matches the Hamiltonian type.
        """
        print("JOONHO here abstract function for construct VHS")

    @plum.dispatch
    def construct_VHS(
        self, hamiltonian: GenericRealISDFChunked, xshifted: xp.ndarray
    ) -> xp.ndarray:
        """Construct ``V_HS`` from ISDF factors distributed over MPI ranks.

        Each rank holds a chunk of the Cholesky/ISDF factors; partial products are
        passed around the shared communicator in a ring until every rank has
        accumulated the full one-body operator for its walkers.

        Parameters
        ----------
        hamiltonian : :class:`~ipie.hamiltonians.chunked_isdf.GenericRealISDFChunked`
            Chunked ISDF Hamiltonian.
        xshifted : :class:`numpy.ndarray`
            Force-bias shifted auxiliary fields, shape ``(nfields, nwalkers)``.

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            One-body operators, one per walker.
        """
        if isinstance(hamiltonian, GenericRealISDFChunked):
            xshifted_send = xshifted.copy()
            xshifted_recv = xp.zeros_like(xshifted)

            idxs = hamiltonian.chol_idxs_chunk
            cholM_chunk = hamiltonian.cholM_chunk
            Lx_send_real = cholM_chunk.dot(xshifted[idxs, :].real)
            Lx_send_imag = cholM_chunk.dot(xshifted[idxs, :].imag)
            Lx_recv_real = xp.zeros_like(Lx_send_real)
            Lx_recv_imag = xp.zeros_like(Lx_send_imag)

            srank = self.mpi_handler.scomm.rank
            sender = numpy.where(self.mpi_handler.receivers == srank)[0]

            for _ in range(self.mpi_handler.ssize - 1):
                synchronize()
                self.mpi_handler.scomm.Isend(
                    xshifted_send, dest=self.mpi_handler.receivers[srank], tag=1
                )
                self.mpi_handler.scomm.Isend(
                    Lx_send_real, dest=self.mpi_handler.receivers[srank], tag=2
                )
                self.mpi_handler.scomm.Isend(
                    Lx_send_imag, dest=self.mpi_handler.receivers[srank], tag=3
                )

                req1 = self.mpi_handler.scomm.Irecv(xshifted_recv, source=sender, tag=1)
                req2 = self.mpi_handler.scomm.Irecv(Lx_recv_real, source=sender, tag=2)
                req3 = self.mpi_handler.scomm.Irecv(Lx_recv_imag, source=sender, tag=3)
                req1.wait()
                req2.wait()
                req3.wait()

                self.mpi_handler.scomm.barrier()

                Lx_send_real = cholM_chunk.dot(xshifted_recv[idxs, :].real)
                Lx_send_imag = cholM_chunk.dot(xshifted_recv[idxs, :].imag)
                Lx_send_real += Lx_recv_real
                Lx_send_imag += Lx_recv_imag

                xshifted_send = xshifted_recv.copy()

            synchronize()
            self.mpi_handler.scomm.Isend(
                Lx_send_real, dest=self.mpi_handler.receivers[srank], tag=1
            )
            self.mpi_handler.scomm.Isend(
                Lx_send_imag, dest=self.mpi_handler.receivers[srank], tag=2
            )
            req1 = self.mpi_handler.scomm.Irecv(Lx_recv_real, source=sender, tag=1)
            req2 = self.mpi_handler.scomm.Irecv(Lx_recv_imag, source=sender, tag=2)
            req1.wait()
            req2.wait()
            self.mpi_handler.scomm.barrier()

            nwalkers = xshifted.shape[-1]
            handle = cutensornet.create()
            network_opts = NetworkOptions(
                handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0]
            )
            VHS_real = contract(
                "Pw, Pp, Pr -> wpr",
                Lx_recv_real,
                hamiltonian.cgto,
                hamiltonian.cgto,
                options=network_opts,
            )
            VHS_imag = contract(
                "Pw, Pp, Pr -> wpr",
                Lx_recv_imag,
                hamiltonian.cgto,
                hamiltonian.cgto,
                options=network_opts,
            )
            VHS = xp.zeros((nwalkers, hamiltonian.nbasis, hamiltonian.nbasis), dtype=xp.complex128)
            VHS.real = VHS_real
            VHS.imag = VHS_imag
            VHS = self.isqrt_dt * VHS
            synchronize()
            xp._default_memory_pool.free_all_blocks()
        else:
            raise ValueError("Invalid hamiltonian type")
        return VHS

    def contract_cholM_xshifted_chunked(
        self, hamiltonian: GenericRealISDFChunked, xshifted: xp.ndarray
    ) -> xp.ndarray:
        if isinstance(hamiltonian, GenericRealISDF):
            xshifted_send = xshifted.copy()
            xshifted_recv = xp.zeros_like(xshifted)

            idxs = hamiltonian.chol_idxs_chunk
            cholM_chunk = hamiltonian.cholM_chunk
            Lx_send = cholM_chunk.dot(xshifted[idxs, :].real).astype(xshifted.dtype)
            Lx_send += 1.0j * cholM_chunk.dot(xshifted[idxs, :].imag)
            Lx_recv = xp.zeros_like(Lx_send)

            srank = self.mpi_handler.scomm.rank
            sender = numpy.where(self.mpi_handler.receivers == srank)[0]

            for _ in range(self.mpi_handler.ssize - 1):
                synchronize()
                self.mpi_handler.scomm.Isend(
                    xshifted_send, dest=self.mpi_handler.receivers[srank], tag=1
                )
                self.mpi_handler.scomm.Isend(Lx_send, dest=self.mpi_handler.receivers[srank], tag=2)

                req1 = self.mpi_handler.scomm.Irecv(xshifted_recv, source=sender, tag=1)
                req2 = self.mpi_handler.scomm.Irecv(Lx_recv, source=sender, tag=2)
                req1.wait()
                req2.wait()

                self.mpi_handler.scomm.barrier()

                Lx_send = cholM_chunk.dot(xshifted_recv[idxs, :].real).astype(xshifted_recv.dtype)
                Lx_send += 1.0j * cholM_chunk.dot(xshifted_recv[idxs, :].imag)
                Lx_send += Lx_recv

                xshifted_send = xshifted_recv.copy()

            synchronize()
            self.mpi_handler.scomm.Isend(Lx_send, dest=self.mpi_handler.receivers[srank], tag=1)
            req = self.mpi_handler.scomm.Irecv(Lx_recv, source=sender, tag=1)
            req.wait()
            self.mpi_handler.scomm.barrier()
        elif isinstance(hamiltonian, GenericComplexISDF):
            raise NotImplementedError("Chunked ISDF complex not implemented")
        else:
            raise ValueError("Invalid hamiltonian type")
        return Lx_recv
