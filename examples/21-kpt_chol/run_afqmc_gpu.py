"""Run k-point Cholesky AFQMC with GPU backend."""

import sys

import h5py
import numpy as np

from ipie.config import config
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler, make_splits_displacements
from ipie.walkers.uhf_walkers import UHFWalkers

try:
    import cupy  # noqa: F401
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

config.update_option("use_gpu", True)

GPU_NUMBER_PER_NODE = 1
NMEMBERS = 1
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

xp.cuda.Device(RANK % GPU_NUMBER_PER_NODE).use()

NSTEPS = 10
NBLOCKS = 10
TIMESTEP = 0.005
EQ_TIMESTEP = 0.02
EQ_NUM_STEPS_PER_BLOCK = 50
EQ_BLOCKS = 0
RNG_SEED = 0

FILENAME = "afqmc_C_311_dz_chol.h5"
with h5py.File(FILENAME, "r") as fa:
    e0 = fa["e0"][()]
    hcore = fa["hcore"][()]
    kpoints = fa["kpoints"][()]
if RANK == 0:
    print("finished reading hcore")
srank = RANK % NMEMBERS

handler = MPIHandler(nmembers=NMEMBERS)
print(f"rank: {RANK}, srank: {srank}")

num_basis = hcore.shape[-1]
with h5py.File(FILENAME, "r") as fa:
    chol_chunk = fa["chol"][()]

chunked_chols = chol_chunk.shape[0]
num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)

split_size = make_splits_displacements(num_chol, NMEMBERS)[0]
assert chunked_chols == split_size[srank]

neleca, nelecb = (4, 4)
system = Generic(nelec=(neleca, nelecb))
ham = KptComplexCholChunked(
    np.array([hcore, hcore]),
    kpoints,
    chol=None,
    chol_chunk=chol_chunk,
    ecore=e0,
    handler=handler,
)

num_basis = ham.nbasis
nk = ham.nk
if RANK == 0:
    print(f"# num_basis: {num_basis}, nk: {nk}, neleca: {neleca}, nelecb: {nelecb}")
psi_a = np.zeros((nk, num_basis, neleca), dtype=np.complex128)
psi_b = np.zeros((nk, num_basis, nelecb), dtype=np.complex128)
phi_a = np.zeros((nk, num_basis, nk, neleca), dtype=np.complex128)
phi_b = np.zeros((nk, num_basis, nk, nelecb), dtype=np.complex128)

for ik in range(nk):
    psi_a[ik] = np.eye(num_basis, neleca, dtype=np.complex128)
    psi_b[ik] = np.eye(num_basis, nelecb, dtype=np.complex128)

for ik1 in range(nk):
    phi_a[ik1, :, ik1, :] = np.eye(num_basis, neleca, dtype=np.complex128)
    phi_b[ik1, :, ik1, :] = np.eye(num_basis, nelecb, dtype=np.complex128)

phia = phi_a.reshape(nk * num_basis, nk * neleca)
phib = phi_b.reshape(nk * num_basis, nk * nelecb)

trial = KptSingleDet(
    np.concatenate([psi_a, psi_b], axis=2),
    nk,
    (neleca, nelecb),
    num_basis,
    handler=handler,
)
trial.build()
trial.half_rotate(ham)

NUM_WALKERS = 10
walkers = UHFWalkers(
    np.hstack([phia, phib]),
    nk * system.nup,
    nk * system.ndown,
    nk * ham.nbasis,
    NUM_WALKERS,
    mpi_handler=handler,
)
walkers.build(trial)
walkers.rhf = True

afqmc = AFQMC.build(
    (neleca, nelecb),
    ham,
    trial,
    walkers,
    NUM_WALKERS,
    RNG_SEED,
    NSTEPS,
    NBLOCKS,
    TIMESTEP,
    eq_timestep=EQ_TIMESTEP,
    eq_num_steps_per_block=EQ_NUM_STEPS_PER_BLOCK,
    num_eq_blocks=EQ_BLOCKS,
    stabilize_freq=5,
    pop_control_freq=5,
    mpi_handler=handler,
    verbose=True,
)
afqmc.run()
afqmc.finalise(verbose=True)
