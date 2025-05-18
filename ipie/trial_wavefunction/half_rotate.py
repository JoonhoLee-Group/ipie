from typing import Tuple

import numpy as np

from ipie.hamiltonians.generic import Generic, GenericComplexChol, GenericRealChol
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.mpi import get_shared_array


def half_rotate_generic(
    trial: TrialWavefunctionBase,
    hamiltonian: Generic,
    comm,
    orbsa: np.ndarray,
    orbsb: np.ndarray,
    ndets: int = 1,
    verbose: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if verbose:
        print("# Constructing half rotated Cholesky vectors.")
    if len(orbsa.shape) == 3:
        assert len(orbsb.shape) == 3
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]
        if trial.verbose:
            print(f"# Shape of alpha half-rotated Cholesky: {ndets, nchol, na * M}")
            print(f"# Shape of beta half-rotated Cholesky: {ndets, nchol, nb * M}")

        chol = hamiltonian.chol.reshape((M, M, nchol))

        shape_a = (ndets, nchol, (M * na))
        shape_b = (ndets, nchol, (M * nb))

        ctype = hamiltonian.chol.dtype
        ptype = orbsa.dtype
        integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype
        if isinstance(hamiltonian, GenericComplexChol):
            cholbar = chol.transpose(1, 0, 2).conj().copy()
            A = hamiltonian.A.reshape((M, M, nchol))
            B = hamiltonian.B.reshape((M, M, nchol))
            rchola = [get_shared_array(comm, shape_a, integral_type) for i in range(4)]
            rcholb = [get_shared_array(comm, shape_b, integral_type) for i in range(4)]
        elif isinstance(hamiltonian, GenericRealChol):
            rchola = [get_shared_array(comm, shape_a, integral_type)]
            rcholb = [get_shared_array(comm, shape_b, integral_type)]

        rH1a = get_shared_array(comm, (ndets, na, M), integral_type)
        rH1b = get_shared_array(comm, (ndets, nb, M), integral_type)

        rH1a[:] = np.einsum("Jpi,pq->Jiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
        rH1b[:] = np.einsum("Jpi,pq->Jiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

        if verbose:
            print("# Half-Rotating Cholesky for determinant.")
        # start = i*M*(na+nb)
        start_a = 0  # determinant loops
        start_b = 0
        compute = True
        # Distribute amongst MPI tasks on this node.
        if comm is not None:
            nwork_per_thread = hamiltonian.nchol // comm.size
            if nwork_per_thread == 0:
                start_n = 0
                end_n = nchol
                if comm.rank != 0:
                    # Just run on root processor if problem too small.
                    compute = False
            else:
                start_n = comm.rank * nwork_per_thread  # Cholesky work split
                end_n = (comm.rank + 1) * nwork_per_thread
                if comm.rank == comm.size - 1:
                    end_n = nchol
        else:
            start_n = 0
            end_n = hamiltonian.nchol

        nchol_loc = end_n - start_n
        if compute:
            if isinstance(hamiltonian, GenericComplexChol):
                L = [chol, cholbar, A, B]
            elif isinstance(hamiltonian, GenericRealChol):
                L = [chol]

            for i in range(len(L)):
                # Investigate whether these einsums are fast in the future
                rup = np.einsum(
                    "Jmi,mnx->Jxin",
                    orbsa.conj(),
                    L[i][:, :, start_n:end_n],
                    optimize=True,
                )
                rup = rup.reshape((ndets, nchol_loc, na * M))
                rdn = np.einsum(
                    "Jmi,mnx->Jxin",
                    orbsb.conj(),
                    L[i][:, :, start_n:end_n],
                    optimize=True,
                )
                rdn = rdn.reshape((ndets, nchol_loc, nb * M))
                rchola[i][:, start_n:end_n, start_a : start_a + M * na] = rup[:]
                rcholb[i][:, start_n:end_n, start_b : start_b + M * nb] = rdn[:]

        if comm is not None:
            comm.barrier()

        if isinstance(hamiltonian, GenericRealChol):
            rchola = rchola[0]
            rcholb = rcholb[0]
    else:
        assert len(orbsa.shape) == 4 #(ndets, nk, nbsf, nocc)
        assert len(orbsb.shape) == 4
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        nk = orbsa.shape[1]
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]
        if isinstance(hamiltonian, KptComplexChol):
            if trial.verbose:
                print(f"# Shape of alpha half-rotated Cholesky: {ndets, nchol, nk, na, nk, M}")
                print(f"# Shape of beta half-rotated Cholesky: {ndets, nchol, nk, nb, nk, M}")

            chol = hamiltonian.chol

            shape_a = (ndets, nchol, nk, na, nk, M)
            shape_b = (ndets, nchol, nk, nb, nk, M)

            ctype = hamiltonian.chol.dtype
            ptype = orbsa.dtype
            integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype

            rchola = get_shared_array(comm, shape_a, integral_type)
            rcholb = get_shared_array(comm, shape_b, integral_type)

            rH1a = get_shared_array(comm, (ndets, nk, na, M), integral_type)
            rH1b = get_shared_array(comm, (ndets, nk, nb, M), integral_type)

            rH1a[:] = np.einsum("Jkpi,kpq->Jkiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
            rH1b[:] = np.einsum("Jkpi,kpq->Jkiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

            if verbose:
                print("# Half-Rotating Cholesky for determinant.")
            # start = i*M*(na+nb)
            start_a = 0  # determinant loops
            start_b = 0
            compute = True
            # Distribute amongst MPI tasks on this node.
            if comm is not None:
                nwork_per_thread = hamiltonian.nchol // comm.size
                if nwork_per_thread == 0:
                    start_n = 0
                    end_n = nchol
                    if comm.rank != 0:
                        # Just run on root processor if problem too small.
                        compute = False
                else:
                    start_n = comm.rank * nwork_per_thread
                    end_n = (comm.rank + 1) * nwork_per_thread
                    if comm.rank == comm.size - 1:
                        end_n = nchol
            else:
                start_n = 0
                end_n = hamiltonian.nchol

            nchol_loc = end_n - start_n
            if compute:
                # Investigate whether these einsums are fast in the future
                rup = np.einsum(
                    "Jkpi,Xkpqr->JXkiqr",
                    orbsa.conj(),
                    chol[start_n:end_n, :, :, :, :],
                    optimize=True,
                )
                rdn = np.einsum(
                    "Jkpi,Xkpqr->JXkiqr",
                    orbsb.conj(),
                    chol[start_n:end_n, :, :, :, :],
                    optimize=True,
                )
                rchola[:, start_n:end_n, :, :, :, :] = rup[:]
                rcholb[:, start_n:end_n, :, :, :, :] = rdn[:]
            if comm is not None:
                comm.barrier()
        elif isinstance(hamiltonian, KptComplexCholSymm):
            unique_nk = hamiltonian.unique_nk
            if trial.verbose:
                print(f"# Shape of alpha half-rotated Cholesky: {ndets, unique_nk, nk, na, nchol, M}")
                print(f"# Shape of beta half-rotated Cholesky: {ndets, unique_nk, nk, nb, nchol, M}")

            chol = hamiltonian.chol

            shape_a = (ndets, unique_nk, nk, na, nchol, M)
            shape_bara = (ndets, unique_nk, nk, M, nchol, na)
            shape_b = (ndets, unique_nk, nk, nb, nchol, M)
            shape_barb = (ndets, unique_nk, nk, M, nchol, nb)

            ctype = hamiltonian.chol.dtype
            ptype = orbsa.dtype
            integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype

            rchola = get_shared_array(comm, shape_a, integral_type)
            rcholbara = get_shared_array(comm, shape_bara, integral_type)
            rcholb = get_shared_array(comm, shape_b, integral_type)
            rcholbarb = get_shared_array(comm, shape_barb, integral_type)

            rH1a = get_shared_array(comm, (ndets, nk, na, M), integral_type)
            rH1b = get_shared_array(comm, (ndets, nk, nb, M), integral_type)

            rH1a[:] = np.einsum("Jkpi,kpq->Jkiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
            rH1b[:] = np.einsum("Jkpi,kpq->Jkiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

            if verbose:
                print("# Half-Rotating Cholesky for determinant.")
            # start = i*M*(na+nb)
            start_a = 0  # determinant loops
            start_b = 0
            compute = True
            # Distribute amongst MPI tasks on this node.
            if comm is not None:
                nwork_per_thread = hamiltonian.nchol // comm.size
                if nwork_per_thread == 0:
                    start_n = 0
                    end_n = nchol
                    if comm.rank != 0:
                        # Just run on root processor if problem too small.
                        compute = False
                else:
                    start_n = comm.rank * nwork_per_thread
                    end_n = (comm.rank + 1) * nwork_per_thread
                    if comm.rank == comm.size - 1:
                        end_n = nchol
            else:
                start_n = 0
                end_n = hamiltonian.nchol

            nchol_loc = end_n - start_n
            if compute:
                # Investigate whether these einsums are fast in the future
                rup = np.einsum(
                    "Jkpi,Xkpqr->JqkiXr",
                    orbsa.conj(),
                    chol[start_n:end_n, :, :, :, :],
                    optimize=True,
                )
                rdn = np.einsum(
                    "Jkpi,Xkpqr->JqkiXr",
                    orbsb.conj(),
                    chol[start_n:end_n, :, :, :, :],
                    optimize=True,
                )
                rchola[:, :, :, :, start_n:end_n, :] = rup[:]
                rcholb[:, :, :, :, start_n:end_n, :] = rdn[:]
                for iq in range(hamiltonian.unique_nk):
                    iq_real = hamiltonian.unique_k[iq]
                    ikpq = hamiltonian.ikpq_mat[iq_real]
                    rbarup = np.einsum(
                        "Jkri, Xkpr -> JkpXi",
                        orbsa[:, ikpq, :, :].conj(),
                        chol[start_n:end_n, :, :, iq, :].conj(),
                        optimize=True,
                    )
                    rbardn = np.einsum(
                        "Jkri, Xkpr -> JkpXi",
                        orbsb[:, ikpq, :, :].conj(),
                        chol[start_n:end_n, :, :, iq, :].conj(),
                    )
                    rcholbara[:, iq, :, :, start_n:end_n, :] = rbarup[:]
                    rcholbarb[:, iq, :, :, start_n:end_n, :] = rbardn[:]                    
            if comm is not None:
                comm.barrier()

            return (rH1a, rH1b), (rchola, rcholb, rcholbara, rcholbarb)

    # storing intermediates for correlation energy
    return (rH1a, rH1b), (rchola, rcholb)


def half_rotate_chunked(
    trial: TrialWavefunctionBase,
    hamiltonian: Generic,
    comm,
    orbsa: np.ndarray,
    orbsb: np.ndarray,
    ndets: int = 1,
    verbose: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    handler = trial.handler
    if verbose:
        print("# Constructing half rotated Cholesky vectors.")
    if len(orbsa.shape) == 3:
        assert len(orbsb.shape) == 3
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]
        if trial.verbose:
            print(f"# Shape of alpha half-rotated Cholesky: {ndets, nchol, na * M}")
            print(f"# Shape of beta half-rotated Cholesky: {ndets, nchol, nb * M}")

        chol_chunk = hamiltonian.chol_chunk.reshape((M, M, -1))
        ctype = hamiltonian.chol_chunk.dtype
        ptype = orbsa.dtype
        integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype
        if isinstance(hamiltonian, GenericComplexChol) or isinstance(hamiltonian, GenericRealChol):
            raise NotImplementedError
        elif isinstance(hamiltonian, GenericRealCholChunked):
            rchola_chunk = [np.zeros((ndets, hamiltonian.nchol_chunk, (M * na)), dtype=integral_type)]
            rcholb_chunk = [np.zeros((ndets, hamiltonian.nchol_chunk, (M * nb)), dtype=integral_type)]
        rH1a = np.einsum("Jpi,pq->Jiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
        rH1b = np.einsum("Jpi,pq->Jiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

        if verbose:
            print("# Half-Rotating Cholesky for determinant.")
        # start = i*M*(na+nb)
        start_a = 0  # determinant loops
        start_b = 0
        compute = True
        # Distribute amongst MPI tasks on this node.
        if comm is not None:
            nwork_per_thread = hamiltonian.nchol // comm.size
            if nwork_per_thread == 0:
                start_n = 0
                end_n = nchol
                if comm.rank != 0:
                    # Just run on root processor if problem too small.
                    compute = False
            else:
                start_n = comm.rank * nwork_per_thread  # Cholesky work split
                end_n = (comm.rank + 1) * nwork_per_thread
                if comm.rank == comm.size - 1:
                    end_n = nchol
        else:
            start_n = 0
            end_n = hamiltonian.nchol

        start_n = hamiltonian.chunk_displacements[handler.srank]
        end_n = hamiltonian.chunk_displacements[handler.srank + 1]

        nchol_loc = end_n - start_n
        if compute:
            # Investigate whether these einsums are fast in the future
            rup = np.einsum(
                "Jmi,mnx->Jxin",
                orbsa.conj(),
                chol_chunk,
                optimize=True,
            )
            rup = rup.reshape((ndets, nchol_loc, na * M))
            rdn = np.einsum(
                "Jmi,mnx->Jxin",
                orbsb.conj(),
                chol_chunk,
                optimize=True,
            )
            rdn = rdn.reshape((ndets, nchol_loc, nb * M))
            rchola_chunk[0][:, :, start_a : start_a + M * na] = rup[:]
            rcholb_chunk[0][:, :, start_b : start_b + M * nb] = rdn[:]

        if comm is not None:
            comm.barrier()

        if isinstance(hamiltonian, GenericRealCholChunked):
            rchola = rchola_chunk[0]
            rcholb = rcholb_chunk[0]
    else:
        assert len(orbsa.shape) == 4
        assert len(orbsb.shape) == 4
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        nk = orbsa.shape[1]
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]
        if isinstance(hamiltonian, KptComplexChol) or isinstance(hamiltonian, KptComplexCholSymm):
            raise NotImplementedError
        elif isinstance(hamiltonian, KptComplexCholChunked):
            unique_nk = hamiltonian.unique_nk
            if trial.verbose:
                print(f"# Shape of alpha half-rotated Cholesky: {ndets, unique_nk, nk, na, nchol, M}")
                print(f"# Shape of beta half-rotated Cholesky: {ndets, unique_nk, nk, nb, nchol, M}")

            chol_chunk = hamiltonian.chol_chunk.reshape(-1, nk, M, unique_nk, M)

            ctype = hamiltonian.chol_chunk.dtype
            ptype = orbsa.dtype
            integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype

            rchola_chunk = [np.zeros((ndets, unique_nk, nk, na, hamiltonian.nchol_chunk, M), dtype=integral_type)]
            rcholbara_chunk = [np.zeros((ndets, unique_nk, nk, M, hamiltonian.nchol_chunk, na), dtype=integral_type)]
            if nb > 0:
                rcholb_chunk = [np.zeros((ndets, unique_nk, nk, nb, hamiltonian.nchol_chunk, M), dtype=integral_type)]
                rcholbarb_chunk = [np.zeros((ndets, unique_nk, nk, M, hamiltonian.nchol_chunk, nb), dtype=integral_type)]
            else:
                rcholb_chunk = [None]
                rcholbarb_chunk = [None]


            rH1a = np.einsum("Jkpi,kpq->Jkiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
            if nb > 0:
                rH1b = np.einsum("Jkpi,kpq->Jkiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)
            else:
                rH1b = None

            if verbose:
                print("# Half-Rotating Cholesky for determinant.")
            # start = i*M*(na+nb)
            start_a = 0  # determinant loops
            start_b = 0
            compute = True
            # Distribute amongst MPI tasks on this node.
            if comm is not None:
                nwork_per_thread = hamiltonian.nchol // comm.size
                if nwork_per_thread == 0:
                    start_n = 0
                    end_n = nchol
                    if comm.rank != 0:
                        # Just run on root processor if problem too small.
                        compute = False
                else:
                    start_n = comm.rank * nwork_per_thread
                    end_n = (comm.rank + 1) * nwork_per_thread
                    if comm.rank == comm.size - 1:
                        end_n = nchol
            else:
                start_n = 0
                end_n = hamiltonian.nchol

            start_n = hamiltonian.chunk_displacements[handler.srank]
            end_n = hamiltonian.chunk_displacements[handler.srank + 1]

            nchol_loc = end_n - start_n
            if compute:
                # Investigate whether these einsums are fast in the future
                rup = np.einsum(
                    "Jkpi,Xkpqr->JqkiXr",
                    orbsa.conj(),
                    chol_chunk,
                    optimize=True,
                )
                if nb > 0:
                    rdn = np.einsum(
                        "Jkpi,Xkpqr->JqkiXr",
                        orbsb.conj(),
                        chol_chunk,
                        optimize=True,
                    )
                rchola_chunk[0][:] = rup[:]
                if nb > 0:
                    rcholb_chunk[0][:] = rdn[:]
                for iq in range(hamiltonian.unique_nk):
                    iq_real = hamiltonian.unique_k[iq]
                    ikpq = hamiltonian.ikpq_mat[iq_real]
                    rbarup = np.einsum(
                        "Jkri, Xkpr -> JkpXi",
                        orbsa[:, ikpq, :, :].conj(),
                        chol_chunk[:, :, :, iq, :].conj(),
                        optimize=True,
                    )
                    if nb > 0:
                        rbardn = np.einsum(
                            "Jkri, Xkpr -> JkpXi",
                            orbsb[:, ikpq, :, :].conj(),
                            chol_chunk[:, :, :, iq, :].conj(),
                        )
                    rcholbara_chunk[0][:, iq, :, :, :, :] = rbarup[:]
                    if nb > 0:
                        rcholbarb_chunk[0][:, iq, :, :, :, :] = rbardn[:]                    
            if comm is not None:
                comm.barrier()

            if isinstance(hamiltonian, KptComplexCholChunked):
                rchola = rchola_chunk[0]
                rcholb = rcholb_chunk[0]
                rcholbara = rcholbara_chunk[0]
                rcholbarb = rcholbarb_chunk[0]

            return (rH1a, rH1b), (rchola, rcholb, rcholbara, rcholbarb)

    # storing intermediates for correlation energy
    return (rH1a, rH1b), (rchola, rcholb)


def half_rotate_isdf(trial: TrialWavefunctionBase,
    hamiltonian: Generic,
    comm,
    orbsa: np.ndarray,
    orbsb: np.ndarray,
    ndets: int = 1,
    verbose: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if len(orbsa.shape) == 3:
        # molecular/gamma pt case
        assert len(orbsb.shape) == 3
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]

        rH1a = np.einsum("Jpi,pq->Jiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
        rH1b = np.einsum("Jpi,pq->Jiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

        # now rotate the cgtos
        cgto = hamiltonian.cgto # [k, P, p]
        rot_cgtoa = np.einsum('Jpi, Pp -> JPi', orbsa, cgto, optimize=True)
        rot_cgtob = np.einsum('Jpi, Pp -> JPi', orbsb, cgto, optimize=True)
        
    else:
        assert len(orbsa.shape) == 4 #(ndets, nk, nbsf, nocc)
        assert len(orbsb.shape) == 4
        assert orbsa.shape[0] == ndets
        assert orbsb.shape[0] == ndets
        M = hamiltonian.nbasis
        nk = orbsa.shape[1]
        na = orbsa.shape[-1]
        nb = orbsb.shape[-1]
        assert isinstance(hamiltonian, KptISDF)

        # ctype = hamiltonian.cholM.dtype
        # ptype = orbsa.dtype
        # integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype

        # rH1a = get_shared_array(comm, (ndets, nk, na, M), integral_type)
        # rH1b = get_shared_array(comm, (ndets, nk, nb, M), integral_type)

        rH1a = np.einsum("Jkpi,kpq->Jkiq", orbsa.conj(), hamiltonian.H1[0], optimize=True)
        rH1b = np.einsum("Jkpi,kpq->Jkiq", orbsb.conj(), hamiltonian.H1[1], optimize=True)

        # now rotate the cgtos
        cgto = hamiltonian.cgto # [k, P, p]
        rot_cgtoa = np.einsum('Jkpi, kPp -> JkPi', orbsa, cgto, optimize=True)
        rot_cgtob = np.einsum('Jkpi, kPp -> JkPi', orbsb, cgto, optimize=True)

    return (rH1a, rH1b), (rot_cgtoa, rot_cgtob)

