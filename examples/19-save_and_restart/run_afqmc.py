"""Example demonstrating how to save and restart AFQMC calculations."""

import numpy as np
from pyscf import gto, scf

from ipie.analysis.extraction import extract_observable
from ipie.estimators.energy import EnergyEstimator, local_energy_batch
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.from_pyscf import generate_hamiltonian, generate_wavefunction_from_mo_coeff

# try:
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
# except ImportError:
from ipie.qmc.comm import FakeComm
comm = FakeComm()

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.RHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()


class EnergyEstimatorDumpWalkers(EnergyEstimator):
    """Energy estimator that also dumps walker configurations periodically."""
    
    def __init__(
        self,
        comm=None,
        qmc=None,
        system=None,
        ham=None,
        trial=None,
        verbose=False,
    ):
        super().__init__(
            system=system,
            ham=ham,
            trial=trial,
        )
        self.nblocks = 0

    def compute_estimator(self, system, walkers, hamiltonian, trial):
        # set as testing purposes. change to larger values for production
        dump_freq = 10
        dump_block_init = 10
        dump_block_end = 10
        
        trial.calc_greens_function(walkers)
        
        # the dumping condition can be adjusted as needed
        if (self.nblocks % dump_freq == 0 and 
            self.nblocks >= dump_block_init and 
            self.nblocks <= dump_block_end):
            walkers.write_walkers_batch(comm)
            
        energy = local_energy_batch(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = np.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = np.sum(walkers.weight)
        self._data["E1Body"] = np.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = np.sum(walkers.weight * energy[:, 2].real)

        self.nblocks += 1
        return self.data


# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World

# Now let's build our custom AFQMC algorithm
num_walkers = 100
num_steps_per_block = 25
num_blocks = 10
timestep = 0.005


# 1. Build Hamiltonian
ham = generate_hamiltonian(
    mol,
    mf.mo_coeff,
    mf.get_hcore(),
    mf.mo_coeff,  # should be optional
)

# 2. Build trial wavefunction
orbs = generate_wavefunction_from_mo_coeff(
    mf.mo_coeff,
    mf.mo_occ,
    mf.mo_coeff,  # Make optional argument
    mol.nelec,
)
num_basis = mf.mo_coeff[0].shape[-1]
trial = SingleDet(
    np.hstack([orbs, orbs]),
    mol.nelec,
    num_basis,
)
trial.build()
trial.half_rotate(ham)

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
add_est = {"energy": EnergyEstimatorDumpWalkers(system=Generic(mol.nelec), ham=ham, trial=trial)}
afqmc.run(additional_estimators=add_est)

qmc_data = extract_observable(afqmc.estimators.filename, "energy")
etotal_last = qmc_data["ETotal"][num_blocks]

# read walkers
afqmc.walkers.read_walkers_batch(trial, comm)
enum, edeom = afqmc.estimators["energy"].compute_estimator(afqmc.system, afqmc.walkers, ham, trial)[:2]
assert np.allclose(enum / edeom, etotal_last)