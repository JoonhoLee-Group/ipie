import sys
import h5py
import numpy

from ipie.hamiltonians.generic_chunked import GenericRealCholChunked as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler

import os
from ipie.utils.backend import arraylib as xp
from pyscf import gto

try:
    import cupy
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

from chunked_chol import *

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)

from ipie.config import config
config.update_option("use_gpu", True)

gpu_number_per_node = 4
nmembers = 4
gpu_id = MPI.COMM_WORLD.rank % gpu_number_per_node
xp.cuda.Device(gpu_id).use()

comm = MPI.COMM_WORLD
num_walkers = 1224 // comm.size
nsteps = 25
nblocks = 100
timestep = 0.005
rng_seed = None


with h5py.File("hamiltonian.h5") as fa:
    e0 = fa["e0"][()]
    hcore = fa["hcore"][()]

rank = comm.Get_rank()
size = comm.Get_size()
srank = rank % nmembers

from ipie.utils.mpi import MPIHandler, make_splits_displacements
handler = MPIHandler(nmembers=nmembers)

from ipie.utils.pack_numba import pack_cholesky

num_basis = hcore.shape[-1]
with h5py.File(f"chol_{srank}.h5") as fa:
    chol_chunk = fa["chol"][()]

chunked_chols = chol_chunk.shape[-1]
num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)

chol_chunk_view = chol_chunk.reshape((num_basis, num_basis, -1))
cp_shape = (num_basis * (num_basis + 1) // 2, chol_chunk_view.shape[-1])
chol_packed_chunk = numpy.zeros(cp_shape, dtype=chol_chunk_view.dtype)
sym_idx = numpy.triu_indices(num_basis)
pack_cholesky(sym_idx[0], sym_idx[1], chol_packed_chunk, chol_chunk_view)
del chol_chunk_view

split_size = make_splits_displacements(num_chol, nmembers)[0]
assert chunked_chols == split_size[srank]

with h5py.File("wavefunction.h5") as fa:
    phi0a = fa["phi0_alpha"][()]
    psiT = fa["psi_T_alpha"][()]


num_basis = hcore.shape[-1]
mol_nelec = mol.nelec
system = Generic(nelec=mol_nelec)
ham = HamGeneric(
    numpy.array([hcore, hcore]),
    None,
    chol_chunk,
    chol_packed_chunk,
    e0, handler
)
ham.nchol = num_chol
ham.handler = handler

trial = SingleDet(numpy.hstack([psiT, psiT]), mol_nelec, num_basis, handler)
trial.build()
trial.half_rotate(ham)

from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.utils.mpi import MPIHandler
walkers = UHFWalkers(numpy.hstack([phi0a, phi0a]), system.nup, system.ndown, ham.nbasis, num_walkers, mpi_handler=handler)
walkers.build(trial)

afqmc = AFQMC.build(
    mol_nelec,
    ham,
    trial,
    walkers,
    num_walkers,
    rng_seed,
    nsteps,
    nblocks,
    timestep,
    mpi_handler=handler)


afqmc.run()
afqmc.finalise(verbose=True)
