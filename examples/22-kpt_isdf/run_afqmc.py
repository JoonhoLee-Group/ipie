"""Run a k-point ISDF AFQMC calculation."""

import sys

import h5py
import numpy as np

from ipie.config import config
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers

try:
    import cupy  # noqa: F401
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

config.update_option("use_gpu", True)

GPU_NUMBER_PER_NODE = 4
NMEMBERS = 1
comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member,no-member
RANK = comm.Get_rank()

xp.cuda.Device(RANK % GPU_NUMBER_PER_NODE).use()

NSTEPS = 10
NBLOCKS = 10
TIMESTEP = 0.005
EQ_TIMESTEP = 0.02
EQ_NUM_STEPS_PER_BLOCK = 50
EQ_BLOCKS = 0
RNG_SEED = 0

FILENAME = "afqmc_C_311_dz_isdf.h5"
with h5py.File(FILENAME, "r") as fa:
    e0 = np.asarray(fa["e0"][()])
    hcore = np.asarray(fa["hcore"][()])
    kpoints = np.asarray(fa["kpoints"][()])
    cgto = np.asarray(fa["cgto"][()])
    mpq = np.asarray(fa["MPQ"][()])

# The integral h5 file provided should contain the following fields (all in MO/OAO basis):
# hcore: the one-body Hamiltonian, shape (nk, nbasis, nbasis)
# MPQ: the ISDF vectors, shape (nunique_k, nisdf, nisdf)
# cgto: Bloch orbitals on the ISDF grid, shape (nk, nisdf, nbasis)
# e0: the constant term in the Hamiltonian, shape ()
# kpoints: the k-points in fractional coordinates, shape (nk, 3)
# unique k is the set of k-points that are unique under inversion symmetry, and the internal order is (Sset, Qplus), where Sset is the set of k-points that are invariant under inversion symmetry, and Qplus is the set of k-points that are not invariant under inversion symmetry but can be paired with another k-point in a set that has no intersection with Qplus to form a pair (k, -k). It should be obtained with the find_self_inverse_set and find_Qplus functions in ipie.utils.kpt_conv.

if RANK == 0:
    print("finished reading hcore")

handler = MPIHandler(nmembers=NMEMBERS)

num_basis = hcore.shape[-1]
chol_m = np.linalg.cholesky(mpq)

neleca, nelecb = (4, 4)
system = Generic(nelec=(neleca, nelecb))
ham = KptISDF(np.array([hcore, hcore]), mpq, chol_m, cgto, kpoints, e0)

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
