from ipie.systems.generic import Generic
from pyscf import gto, scf, lo
from ipie.utils.from_pyscf import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC
import numpy
import mpi4py.MPI as MPI
import h5py

comm = MPI.COMM_WORLD

nocca = 5
noccb = 5
nelec = nocca + noccb
r0 = 1.75
mol = gto.M(
    atom=[("H", i * r0, 0, 0) for i in range(nelec)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5
)

mf = scf.UHF(mol).run()
mf.chkfile = 'scf.chk'

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability(return_status=True)

s1e = mol.intor("int1e_ovlp_sph")
ao_coeff = lo.orth.lowdin(s1e)

system = Generic(nelec=mol.nelec)

integrals = generate_hamiltonian(
        mol,
        mf.mo_coeff,
        mf.get_hcore(),
        ao_coeff,
        ortho_ao=True,
        chol_cut=1.e-8
    )

num_chol = integrals.chol.shape[1]
num_basis = integrals.nbasis

with h5py.File("generic_integrals.h5", "w") as fa:
    fa["hcore"] = integrals.H1[0]
    fa["LXmn"] = integrals.chol.T.reshape(num_chol, num_basis, num_basis)
    fa["e0"] = integrals.ecore

mu = -10.0

options = {
    "qmc": {
        "dt": 0.01,
        "nwalkers": 1,
        "blocks": 10,
        "nsteps": 10,
        "beta": 0.1,
        "rng_seed": 7,
        "pop_control_freq": 1,
        "stabilise_freq": 10,
        "batched": False
    },
    "walkers": {
        "population_control": "pair_branch"
    },
    "estimators": {
        "mixed": {
            "one_rdm": True
        }
    },
    "trial": {
        "name": "thermal_hartree_fock"
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "generic_integrals.h5",
        "_alt_convention": False,
        "sparse": False,
        "mu": mu
    },
    "system": {
        "name": "Generic",
        "nup": nocca,
        "ndown": noccb,
        "mu": mu
    }
}


afqmc = ThermalAFQMC(comm, options)
afqmc.run(comm=comm)
afqmc.finalise(comm)