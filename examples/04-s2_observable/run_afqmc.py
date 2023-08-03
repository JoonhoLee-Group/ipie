import os

import numpy as np
from pyscf import gto, scf

from ipie.config import MPI
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

mol = gto.M(
    atom=[("H", 3.2 * i, 0, 0) for i in range(2)],
    basis="sto3g",
    verbose=4,
    unit="Angstrom",
)
mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()
mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)

ndown = 1
nup = 1
Ms = (nup - ndown) / 2.0
P = mf.make_rdm1()
Pa = P[0].copy()
Pb = P[1].copy()

S = mol.intor("int1e_ovlp")

Ca = mf.mo_coeff[0]
Cb = mf.mo_coeff[1]

S2ref = ndown + Ms * (Ms + 1.0) - np.trace(Pa @ S @ Pb @ S)
# mf.mulliken_pop(mol)

# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0, ortho_ao=False)

comm = MPI.COMM_WORLD
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.greens_function import greens_function
from ipie.qmc.calc import build_afqmc_driver


class S2Mixed(EstimatorBase):
    def __init__(self, ham):
        # We define a dictionary to contain whatever we want to compute.
        # Note we typically want to separate the numerator and denominator of
        # the estimator
        # We require complex valued buffers for accumulation
        self._data = {
            "S2Numer": np.zeros((1), dtype=np.complex128),
            "S2Denom": np.zeros((1), dtype=np.complex128),
        }
        # We also need to specify the shape of the desired estimator
        self._shape = (1,)
        # Optional but good to know (we can redirect to custom filepath (ascii)
        # and / or print to stdout but we shouldnt do this for non scalar
        # quantities
        self.print_to_stdout = True
        self.ascii_filename = None
        # Must specify that we're dealing with array valued estimator
        self.scalar_estimator = False

    def compute_estimator(self, system, walkers, hamiltonian, trial):
        greens_function(walkers, trial, build_full=True)

        ndown = system.ndown
        nup = system.nup
        Ms = (nup - ndown) / 2.0
        two_body = -np.einsum("wij,wji->w", walkers.Ga, walkers.Gb)
        two_body = two_body * walkers.weight

        denom = np.sum(walkers.weight)
        numer = np.sum(two_body) + denom * (Ms * (Ms + 1) + ndown)

        self["S2Numer"] = numer
        self["S2Denom"] = denom


afqmc = build_afqmc_driver(comm, nelec=mol.nelec, num_walkers_per_task=10, verbosity=-10)
# Let us override the number of blocks to keep it short
afqmc.params.num_blocks = 50
# afqmc.estimators.overwite = True
# We can now add this to the estimator handler object in the afqmc driver
estimators = {"S2": S2Mixed(ham=afqmc.hamiltonian)}
afqmc.run(additional_estimators=estimators)
afqmc.finalise(verbose=True)
# We can extract the qmc data as as a pandas data frame like so
from ipie.analysis.extraction import extract_observable

# Note the 'energy' estimator is always computed.
qmc_data = extract_observable(afqmc.estimators.filename, "S2")
# Should be close to 1
np.testing.assert_almost_equal(S2ref, qmc_data[0, 0].real)
