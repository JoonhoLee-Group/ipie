"""Simple AFQMC script for a 1D Hubbard model constructed from pyscf."""

import numpy
import scipy.linalg
from pyscf import ao2mo, gto, scf

from ipie.hamiltonians.generic import GenericRealChol
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet

mol = gto.M()
n = 2
U = 4
mol.nelectron = n

mf = scf.UHF(mol)
h1 = numpy.zeros((n, n))
for i in range(n-1):
    h1[i, i+1] = h1[i+1, i] = -1.0
h1[n-1, 0] = h1[0, n-1] = -1.0  # PBC

eri = numpy.zeros((n, n, n, n))
for i in range(n):
    eri[i, i, i, i] = U

u, s, vdag = scipy.linalg.svd(eri.reshape(n**2, n**2))
chol = (u @ numpy.diag(s**(1/2)))[:, :n]

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)

dma = numpy.array([[1.0, 0.0], [0.0, 0.0]])
dmb = numpy.array([[0.0, 0.0], [0.0, 1.0]])
dm0 = [dma, dmb]
mf.run(dm0)

# spin up and down
na, nb = mf.nelec
phia = mf.mo_coeff[0][:, :na]
phib = mf.mo_coeff[1][:, :nb]

# Now let's build our custom AFQMC algorithm
num_walkers = 640
num_steps_per_block = 25
num_blocks = 2 # Adjust this in practice
timestep = 0.005

# 1. Build Hamiltonian
ham = GenericRealChol(numpy.array([h1, h1]), chol, 0)

# 2. Build trial wavefunction
trial = SingleDet(
    wavefunction=numpy.hstack([phia, phib]),
    num_elec=mf.nelec,
    num_basis=n
)

trial.build()
trial.half_rotate(ham)

# 3. Run AFQMC
afqmc = AFQMC.build(
    mol.nelec,
    ham,
    trial,
    num_walkers=num_walkers,
    num_steps_per_block=num_steps_per_block,
    num_blocks=num_blocks,
    timestep=timestep,
    seed=59306159,
)
afqmc.run()
