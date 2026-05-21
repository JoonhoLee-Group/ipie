# Author: Maxine Luo <man.luo@mpq.mpg.de>
#         Victor Chen <victor.chen@tum.de>
#

"""Benchmark GPU of ithc AFQMC calculation."""

import sys

import numpy as np
from ipie.utils.backend import arraylib as xp

xp.set_printoptions(suppress=True)
import h5py
from ipie.config import MPI

# try:
#    import cupy
# except ImportError:
#    sys.exit(0)

from ipie.config import config

# config.update_option("use_gpu", True)

from ipie.hamiltonians.generic_ithc import import_Hamiltonian
from ipie.hamiltonians.generic_ithc import GenericITHC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.qmc.afqmc import AFQMC
from pyscf import gto, scf, ao2mo, fci


def generate_random_isometry(nbasis_extended, nbasis, xp=np):
    """
    Generates a random real isometry matrix U.
    """
    random_mat = xp.random.randn(nbasis_extended, nbasis_extended)

    # QR decomposition gives an orthonormal basis (Q)
    q, r = xp.linalg.qr(random_mat)

    # Slice to the requested dimensions
    isometry = q[:nbasis, :]
    return isometry


def generate_random_mat(nbasis, xp=np):
    """
    Generates a random symmetric matrix W
    """
    # Create a random matrix
    A = xp.random.randn(nbasis, nbasis)
    # Ensure it is symmetric: W = A + A.T
    W = A + A.T
    return W


def generate_orthonormal_orbitals(n_basis, n_occ):
    """Generates a random orthonormal set of orbitals."""
    # Random matrix
    rand_mat = xp.random.randn(n_basis, n_occ) + 1.0j * xp.random.randn(n_basis, n_occ)
    # QR decomposition to get orthonormal columns
    q, r = xp.linalg.qr(rand_mat)
    return q


nelec = (10, 10)
nbasis = 80
nbasis_extended = 120

# construct trial wavefunction
na, nb = nelec
psi0a = generate_orthonormal_orbitals(nbasis, na)
psi0b = generate_orthonormal_orbitals(nbasis, nb)

trial = SingleDet(wavefunction=np.hstack((psi0a, psi0b)), num_elec=nelec, num_basis=nbasis)

trial.build()

one_electron_random = generate_random_mat(nbasis)
isometry = generate_random_isometry(nbasis_extended, nbasis)
W = generate_random_mat(nbasis_extended)
hamil = GenericITHC(one_electron_random, isometry, W)

print(nelec)
print(f"Shape of trial.psi0a: {xp.shape(trial.psi0a)}")
print(f"Shape of isometry: {xp.shape(hamil.isometry)}")

num_walkers = 100
num_steps_per_block = 25
num_blocks = 10
timestep = 0.005

trial.half_rotate(hamil)

afqmc = AFQMC.build(
    nelec,
    hamil,
    trial,
    num_walkers=num_walkers,
    num_steps_per_block=num_steps_per_block,
    num_blocks=num_blocks,
    timestep=timestep,
    seed=593061,
    verbose=False,
)
afqmc.run()
afqmc.finalise(verbose=True)
