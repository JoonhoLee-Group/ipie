import sys
import h5py
import numpy

from ipie.hamiltonians.generic_chunked import GenericRealCholChunked as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.hamiltonians.utils import get_hamiltonian

import os
from ipie.utils.backend import arraylib as xp
from pyscf import gto

try:
    from mpi4py import MPI
except ImportError:
    sys.exit(0)


mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)

mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)

shared_comm = get_shared_comm(comm, verbose=True)
ham = get_hamiltonian(dir+"hamiltonian.h5", shared_comm, verbose=True, pack_chol=True)

with h5py.File(dir+"wavefunction.h5") as fa:
    phi0a = fa["phi0_alpha"][()]
    psiT = fa["psi_T_alpha"][()]

num_basis = phi0a.shape[0]
mol_nelec = mol.nelec
system = Generic(nelec=mol_nelec)

trial = SingleDet(numpy.hstack([psiT, psiT]), mol_nelec, num_basis)
trial.build()
trial.half_rotate(ham)

from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.utils.mpi import MPIHandler
walkers = UHFWalkers(numpy.hstack([phi0a, phi0a]), system.nup, system.ndown, ham.nbasis, num_walkers, mpi_handler=MPIHandler())

num_walkers = 1224 // comm.size
nsteps = 25
nblocks = 100
timestep = 0.005
rng_seed = None

afqmc = AFQMC.build(
    mol_nelec,
    ham,
    trial,
    walkers,
    num_walkers,
    rng_seed,
    nsteps,
    nblocks,
    timestep)

afqmc.run()
afqmc.finalise(verbose=True)
