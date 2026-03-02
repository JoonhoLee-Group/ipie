# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

import time

import numpy

from ipie.hamiltonians.generic import construct_h1e_mod, Generic, read_integrals
from ipie.hamiltonians.generic import GenericComplexChol
from ipie.hamiltonians.kpt_hamiltonian import KptComplexCholSymm, read_kpt_integrals
from ipie.utils.mpi import get_shared_array, have_shared_mem
from ipie.utils.pack_numba import pack_cholesky


def get_hamiltonian(filename, scomm, verbose=False, pack_chol=True):
    """Wrapper to select hamiltonian class with integrals in shared memory.

    Parameters
    ----------
    filename : str
        Hamiltonian filename.
    scomm : MPI.COMM_WORLD
        MPI split communicator (shared memory).
    pack_chol : bool
        Only store minimum amount of information required by integrals.
    verbose : bool
        Output verbosity.

    Returns
    -------
    ham : object
        Hamiltonian class.
    """
    start = time.time()
    hcore, chol, _, enuc = get_generic_integrals(filename, comm=scomm, verbose=verbose)
    if verbose:
        print(f"# Time to read integrals: {time.time() - start:.6f}")

    start = time.time()

    nbsf = hcore.shape[-1]
    nchol = chol.shape[-1]
    idx = numpy.triu_indices(nbsf)

    chol = chol.reshape((nbsf, nbsf, nchol))

    shmem = have_shared_mem(scomm)
    if shmem:
        if scomm.rank == 0:
            cp_shape = (nbsf * (nbsf + 1) // 2, nchol)
            dtype = chol.dtype
        else:
            cp_shape = None
            dtype = None

        shape = scomm.bcast(cp_shape, root=0)
        dtype = scomm.bcast(dtype, root=0)
        chol_packed = get_shared_array(scomm, shape, dtype)
        if scomm.rank == 0 and pack_chol:
            pack_cholesky(idx[0], idx[1], chol_packed, chol)
        scomm.Barrier()
        chol_pack_shmem = get_shared_array(scomm, shape, dtype)
        if scomm.rank == 0:
            chol_pack_shmem[:] = chol_packed[:]
    else:
        dtype = chol.dtype
        cp_shape = (nbsf * (nbsf + 1) // 2, nchol)
        chol_packed = numpy.zeros(cp_shape, dtype=dtype)
        if pack_chol:
            pack_cholesky(idx[0], idx[1], chol_packed, chol)

    chol = chol.reshape((nbsf * nbsf, nchol))

    if verbose:
        print(f"# Time to pack Cholesky vectors: {time.time() - start:.6f}")

    if shmem and pack_chol:
        ham = Generic(
            h1e=hcore, chol=chol, ecore=enuc, shmem=True, chol_packed=chol_packed, verbose=verbose
        )
    else:
        ham = Generic(h1e=hcore, chol=chol, ecore=enuc, verbose=verbose)

    return ham


def get_kpt_hamiltonian(filename, scomm, verbose=False):
    """Wrapper to select hamiltonian class with integrals in shared memory.

    Parameters
    ----------
    filename : str
        Hamiltonian filename.
    scomm : MPI.COMM_WORLD
        MPI split communicator (shared memory).
    pack_chol : bool
        Only store minimum amount of information required by integrals.
    verbose : bool
        Output verbosity.

    Returns
    -------
    ham : object
        Hamiltonian class.
    """
    start = time.time()
    hcore, chol, kpts, enuc = get_kpt_integrals(filename, comm=scomm, verbose=verbose)
    if verbose:
        print(f"# Time to read integrals: {time.time() - start:.6f}")

    start = time.time()

    ham = KptComplexCholSymm(h1e=hcore, chol=chol, kpts=kpts, ecore=enuc, verbose=verbose)

    return ham


def get_complex_hamiltonian(filename, scomm, verbose=False):
    """Wrapper to complex hamiltonian class with integrals in shared memory.

    Parameters
    ----------
    filename : str
        Hamiltonian filename.
    scomm : MPI.COMM_WORLD
        MPI split communicator (shared memory).
    pack_chol : bool
        Only store minimum amount of information required by integrals.
    verbose : bool
        Output verbosity.

    Returns
    -------
    ham : object
        Hamiltonian class.
    """
    start = time.time()
    hcore, chol, _, enuc = get_generic_integrals(filename, comm=scomm, verbose=verbose)
    if verbose:
        print(f"# Time to read integrals: {time.time() - start:.6f}")

    start = time.time()

    nbsf = hcore.shape[-1]
    nchol = chol.shape[-1]

    chol = chol.reshape((nbsf, nbsf, nchol))
    assert chol.dtype == numpy.complex128

    shmem = have_shared_mem(scomm)
    if shmem:
        if scomm.rank == 0:
            dtype = chol.dtype
        else:
            dtype = None

        dtype = scomm.bcast(dtype, root=0)

        A = get_shared_array(scomm, chol.shape, dtype)
        B = get_shared_array(scomm, chol.shape, dtype)
        if scomm.rank == 0:
            for x in range(nchol):
                A[:, :, x] = (chol[:, :, x] + chol[:, :, x].T.conj()) / 2.0
                B[:, :, x] = 1.0j * (chol[:, :, x] - chol[:, :, x].T.conj()) / 2.0
        scomm.Barrier()
        A = A.reshape((nbsf * nbsf, nchol))
        B = B.reshape((nbsf * nbsf, nchol))

    chol = chol.reshape((nbsf * nbsf, nchol))

    if verbose:
        print(f"# Time to pack Cholesky vectors: {time.time() - start:.6f}")

    if shmem:
        ham = GenericComplexChol(
            h1e=hcore, chol=chol, ecore=enuc, shmem=True, A=A, B=B, verbose=verbose
        )
    else:
        ham = Generic(h1e=hcore, chol=chol, ecore=enuc, verbose=verbose)

    return ham


def get_generic_integrals(filename, comm=None, verbose=False):
    """Read generic integrals, potentially into shared memory.

    Parameters
    ----------
    filename : string
        File containing 1e- and 2e-integrals.
    comm : MPI communicator
        split communicator. Optional. Default: None.
    verbose : bool
        Write information.

    Returns
    -------
    hcore : :class:`numpy.ndarray`
        One-body hamiltonian.
    chol : :class:`numpy.ndarray`
        Cholesky tensor L[ik,n].
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian following subtraction of normal ordered
        contributions.
    enuc : float
        Core energy.
    """
    shmem = have_shared_mem(comm)
    if verbose:
        print(f"# Have shared memory: {shmem}")
    if shmem:
        if comm.rank == 0:
            hcore, chol, enuc = read_integrals(filename)
            hc_shape = hcore.shape
            ch_shape = chol.shape
            dtype = chol.dtype
        else:
            hc_shape = None
            ch_shape = None
            dtype = None
            enuc = None
        shape = comm.bcast(hc_shape, root=0)
        dtype = comm.bcast(dtype, root=0)
        enuc = comm.bcast(enuc, root=0)
        hcore_shmem = get_shared_array(comm, (2,) + shape, dtype)
        if comm.rank == 0:
            hcore_shmem[0] = hcore[:]
            hcore_shmem[1] = hcore[:]
        comm.Barrier()
        shape = comm.bcast(ch_shape, root=0)
        chol_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            chol_shmem[:] = chol[:]
        comm.Barrier()
        h1e_mod_shmem = get_shared_array(comm, hcore_shmem.shape, dtype)
        if comm.rank == 0:
            construct_h1e_mod(chol_shmem, hcore_shmem, h1e_mod_shmem)
        comm.Barrier()
        return hcore_shmem, chol_shmem, h1e_mod_shmem, enuc
    else:
        hcore, chol, enuc = read_integrals(filename)
        h1 = numpy.array([hcore, hcore])
        h1e_mod = numpy.zeros(h1.shape, dtype=h1.dtype)
        construct_h1e_mod(chol, h1, h1e_mod)
        return h1, chol, h1e_mod, enuc


def get_kpt_integrals(filename, comm=None, verbose=False):
    """Read kpt integrals, potentially into shared memory.

    Parameters
    ----------
    filename : string
        File containing 1e- and 2e-integrals.
    comm : MPI communicator
        split communicator. Optional. Default: None.
    verbose : bool
        Write information.

    Returns
    -------
    hcore : :class:`numpy.ndarray`
        One-body hamiltonian.
    chol : :class:`numpy.ndarray`
        Cholesky tensor L[ik,n].
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian following subtraction of normal ordered
        contributions.
    enuc : float
        Core energy.
    """
    shmem = have_shared_mem(comm)
    if verbose:
        print(f"# Have shared memory: {shmem}")
    if shmem:
        if comm.rank == 0:
            hcore, chol, kpts, enuc = read_kpt_integrals(filename)
            hc_shape = hcore.shape
            ch_shape = chol.shape
            kpt_shape = kpts.shape
            dtype = chol.dtype
        else:
            hc_shape = None
            ch_shape = None
            kpt_shape = None
            dtype = None
            enuc = None
        shape = comm.bcast(hc_shape, root=0)
        dtype = comm.bcast(dtype, root=0)
        enuc = comm.bcast(enuc, root=0)
        hcore_shmem = get_shared_array(comm, (2,) + shape, dtype)
        if comm.rank == 0:
            hcore_shmem[0] = hcore[:]
            hcore_shmem[1] = hcore[:]
        comm.Barrier()
        shape = comm.bcast(kpt_shape, root=0)
        kpts_shmem = get_shared_array(comm, shape, numpy.float64)
        if comm.rank == 0:
            kpts_shmem[:] = kpts[:]
        shape = comm.bcast(ch_shape, root=0)
        chol_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            chol_shmem[:] = chol[:]
        comm.Barrier()
        return hcore_shmem, chol_shmem, kpts_shmem, enuc
    else:
        hcore, chol, kpts, enuc = read_integrals(filename)
        h1 = numpy.array([hcore, hcore])
        return h1, chol, kpts, enuc
