from typing import Tuple

import numpy as np

from ipie.hamiltonians.generic import Generic, GenericComplexChol, GenericRealChol
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
    assert len(orbsa.shape) == 3
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

    # storing intermediates for correlation energy
    return (rH1a, rH1b), (rchola, rcholb)
