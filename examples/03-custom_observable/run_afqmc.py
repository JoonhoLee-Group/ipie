import os

import numpy as np

from pyscf import cc, gto, scf

from mpi4py import MPI

from ipie.hamiltonians.utils import get_hamiltonian
from ipie.qmc.afqmc_batch import AFQMCBatch
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from ipie.utils.mpi import get_shared_comm

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)

comm = MPI.COMM_WORLD

from ipie.qmc.calc import build_afqmc_driver

afqmc = build_afqmc_driver(comm, nelec=mol.nelec)
# Inspect the default qmc options
print(afqmc.qmc)
# Let us override the number of blocks to keep it short
afqmc.qmc.nblocks = 20
afqmc.estimators.overwite = True
afqmc.run(comm=comm)
# We can extract the qmc data as as a pandas data frame like so
from ipie.analysis.extraction import extract_observable

# Note the 'energy' estimator is always computed.
qmc_data = extract_observable(afqmc.estimators.filename, "energy")
print(qmc_data.head())

# We can also add custom estimators via the EstimatorBase base class As a simple
# example let us add an estimator to extract the diagonal of the **mixed
# estimate** of the 1RDM. It is **critically** important you understand the
# limitations of mixed estimates in QMC calculations before proceeding.

from ipie.estimators.estimator_base import EstimatorBase

# Let's define our estimator class
# For interface consistency we pass several objects around like hamiltonian and
# wavefunction but these are not required to be used.
from ipie.estimators.greens_function_batch import greens_function


class Diagonal1RDM(EstimatorBase):
    def __init__(self, ham):

        # We define a dictionary to contain whatever we want to compute.
        # Note we typically want to separate the numerator and denominator of
        # the estimator
        # We require complex valued buffers for accumulation
        self._data = {
            "DiagGNumer": np.zeros((ham.nbasis), dtype=np.complex128),
            "DiagGDenom": np.zeros((1), dtype=np.complex128),
        }
        # We also need to specify the shape of the desired estimator
        self._shape = (ham.nbasis,)
        # Optional but good to know (we can redirect to custom filepath (ascii)
        # and / or print to stdout but we shouldnt do this for non scalar
        # quantities
        self.print_to_stdout = False
        self.ascii_filename = None
        # Must specify that we're dealing with array valued estimator
        self.scalar_estimator = False

    def compute_estimator(self, system, walker_batch, hamiltonian, trial_wavefunction):
        greens_function(walker_batch, trial_wavefunction, build_full=True)
        from ipie.estimators.greens_function_batch import get_greens_function

        numer = np.einsum(
            "w,wii->i", walker_batch.weight, walker_batch.Ga + walker_batch.Gb
        )
        self["DiagGNumer"] = numer
        self["DiagGDenom"] = sum(walker_batch.weight)


afqmc = build_afqmc_driver(comm, nelec=mol.nelec)
# Let us override the number of blocks to keep it short
afqmc.qmc.nblocks = 20
afqmc.estimators.overwite = True
# We can now add this to the estimator handler object in the afqmc driver
afqmc.estimators["diagG"] = Diagonal1RDM(ham=afqmc.hamiltonian)
afqmc.run(comm=comm)
# We can extract the qmc data as as a pandas data frame like so
from ipie.analysis.extraction import extract_observable

# Note the 'energy' estimator is always computed.
qmc_data = extract_observable(afqmc.estimators.filename, "diagG")
# Should be close to 10 (the number of electrons in the system)
print(sum(qmc_data[0]).real)


class Mixed1RDM(EstimatorBase):
    def __init__(self, ham):

        # We define a dictionary to contain whatever we want to compute.
        # Note we typically want to separate the numerator and denominator of
        # the estimator
        # We require complex valued buffers for accumulation
        self._shape = (2, ham.nbasis, ham.nbasis)
        # For multi-dimensional estiamtors we need to flatten data
        self._data = {
            "GNumer": np.zeros((np.prod(self._shape)), dtype=np.complex128),
            "GDenom": np.zeros((1), dtype=np.complex128),
        }
        # We also need to specify the shape of the desired estimator
        # Optional but good to know (we can redirect to custom filepath (ascii)
        # and / or print to stdout but we shouldnt do this for non scalar
        # quantities
        self.print_to_stdout = False
        self.ascii_filename = None
        # Must specify that we're dealing with array valued estimator
        self.scalar_estimator = False

    def compute_estimator(self, system, walker_batch, hamiltonian, trial_wavefunction):
        greens_function(walker_batch, trial_wavefunction, build_full=True)
        from ipie.estimators.greens_function_batch import get_greens_function

        numer = np.array(
            [
                np.einsum("w,wij->ij", walker_batch.weight, walker_batch.Ga),
                np.einsum("w,wij->ij", walker_batch.weight, walker_batch.Gb),
            ]
        )

        # For multidimensional arrays we must flatten the data
        self["GNumer"] = numer.ravel()
        self["GDenom"] = sum(walker_batch.weight)


afqmc = build_afqmc_driver(comm, nelec=mol.nelec)
# Let us override the number of blocks to keep it short
afqmc.qmc.nblocks = 20
afqmc.estimators.overwite = True
# We can now add this to the estimator handler object in the afqmc driver
afqmc.estimators["diagG"] = Diagonal1RDM(ham=afqmc.hamiltonian)
afqmc.estimators["1RDM"] = Mixed1RDM(ham=afqmc.hamiltonian)
afqmc.run(comm=comm)
# We can extract the qmc data as as a pandas data frame like so
from ipie.analysis.extraction import extract_observable

# Note the 'energy' estimator is always computed.
qmc_data = extract_observable(afqmc.estimators.filename, "1RDM")
# Should be close to 10 (the number of electrons in the system)
assert qmc_data.shape == (21, 2, 10, 10)
# Should be close to 5 (alpha part of the RDM).
print(qmc_data[0, 0].trace().real)

# Necessary to test your implementation, in particular that reading / writing is
# happening as expected.
greens_function(afqmc.psi.walkers_batch, afqmc.trial, build_full=True)
weights = afqmc.psi.walkers_batch.weight
refa = np.einsum("w,wij->ij", weights, afqmc.psi.walkers_batch.Ga.copy())
refa /= np.sum(weights)
qmc_data = extract_observable(afqmc.estimators.filename, "1RDM")
assert qmc_data.shape == (21, 2, 10, 10)
assert np.allclose(refa, qmc_data[-1, 0])
