from mpi4py import MPI
import numpy as np
from typing import Tuple
import time

from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.mpi import get_shared_array
from ipie.systems.generic import Generic as SysGeneric
from ipie.hamiltonians.generic import Generic as HamGeneric

def half_rotate_generic(
    trial: TrialWavefunctionBase,
    system: SysGeneric,
    hamiltonian: HamGeneric,
    comm: MPI.COMM_WORLD,
    orbsa: np.ndarray,
    orbsb: np.ndarray,
    ndets: int=1,
    verbose: bool=False,
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
        print("# Shape of alpha half-rotated Cholesky: {}".format((ndets, nchol, na * M)))
        print("# Shape of beta half-rotated Cholesky: {}".format((ndets, nchol, nb * M)))

    chol = hamiltonian.chol_vecs.reshape((M, M, nchol))

    shape_a = (ndets, nchol, (M * na))
    shape_b = (ndets, nchol, (M * nb))


    ctype = hamiltonian.chol_vecs.dtype
    ptype = orbsa.dtype
    integral_type = ctype if ctype.itemsize > ptype.itemsize else ptype
    rchola = get_shared_array(comm, shape_a, integral_type)
    rcholb = get_shared_array(comm, shape_b, integral_type)

    rH1a = get_shared_array(comm, (ndets, na, M), integral_type)
    rH1b = get_shared_array(comm, (ndets, nb, M), integral_type)

    print(orbsa.shape, orbsb.shape,)
    rH1a[:] = np.einsum("Jpi,pq->Jiq", orbsa, hamiltonian.H1[0], optimize=True)
    rH1b[:] = np.einsum("Jpi,pq->Jiq", orbsb, hamiltonian.H1[0], optimize=True)

    start_time = time.time()
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
        # Investigate whether these einsums are fast in the future
        rup = np.einsum(
            "Jmi,mnx->Jxin",
            orbsa.conj(),
            chol[:, :, start_n:end_n],
            optimize=True,
        )
        rup = rup.reshape((ndets, nchol_loc, na * M))
        rdn = np.einsum(
            "Jmi,mnx->Jxin",
            orbsb.conj(),
            chol[:, :, start_n:end_n],
            optimize=True,
        )
        rdn = rdn.reshape((ndets, nchol_loc, nb * M))
        rchola[:, start_n:end_n, start_a : start_a + M * na] = rup[:]
        rcholb[:, start_n:end_n, start_b : start_b + M * nb] = rdn[:]

    if comm is not None:
        comm.barrier()

    # storing intermediates for correlation energy
    return (rH1a, rH1b), (rchola, rcholb)
