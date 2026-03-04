"""Run a k-point Cholesky AFQMC calculation."""

import numpy as np
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    from ipie.qmc.comm import FakeComm
    comm = FakeComm()

from ipie.hamiltonians.utils import get_kpt_hamiltonian
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.mpi import MPIHandler, get_shared_comm
from ipie.walkers.uhf_walkers import UHFWalkers

NSTEPS = 10
NBLOCKS = 10
TIMESTEP = 0.005
EQ_TIMESTEP = 0.02
EQ_NUM_STEPS_PER_BLOCK = 50
EQ_BLOCKS = 0
RNG_SEED = 0

scomm = get_shared_comm(comm, verbose=True)

handler = MPIHandler()

ham = get_kpt_hamiltonian("./afqmc_C_311_dz_chol.h5", scomm, verbose=True)

num_basis = ham.nbasis
nk = ham.nk
neleca, nelecb = (4, 4)
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

system = Generic(nelec=(neleca, nelecb))

trial = KptSingleDet(
    np.concatenate([psi_a, psi_b], axis=2),
    nk,
    (neleca, nelecb),
    num_basis,
)
trial.build()
trial.half_rotate(ham, scomm)

NUM_WALKERS = 10
walkers = UHFWalkers(
    np.hstack([phia, phib]),
    nk * system.nup,
    nk * system.ndown,
    nk * ham.nbasis,
    NUM_WALKERS,
    mpi_handler=handler,
)

RNG_SEED = 0
afqmc = AFQMC.build(
    (neleca, nelecb),
    ham,
    trial,
    walkers=walkers,
    num_walkers=NUM_WALKERS,
    seed=RNG_SEED,
    num_steps_per_block=NSTEPS,
    num_blocks=NBLOCKS,
    timestep=TIMESTEP,
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
