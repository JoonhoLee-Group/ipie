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


def compare_test_data(ref, test):
    comparison = {}
    for k, v in ref.items():
        if k == "sys_info":
            continue
        try:
            comparison[k] = (
                numpy.array(ref[k]),
                numpy.array(test[k]),
                numpy.max(numpy.abs(numpy.array(ref[k]) - numpy.array(test[k]))) < 1e-10,
            )
        except KeyError:
            print(f"# Issue with test data key {k}")
    return comparison

test_name = "generic"
import tempfile
import json
from ipie.analysis.extraction import extract_test_data_hdf5
with tempfile.NamedTemporaryFile() as tmpf:
    options["estimators"]["filename"] = tmpf.name
    afqmc = ThermalAFQMC(comm, options=options, parallel=comm.size > 1, verbose=1)
    afqmc.run(comm=comm)
    afqmc.finalise(comm)
    test_data = extract_test_data_hdf5(tmpf.name)
    with open("reference_data/generic_ref.json", "r") as fa:
        ref_data = json.load(fa)
    comparison = compare_test_data(ref_data, test_data)
    local_err_count = 0
    for k, v in comparison.items():
        if not v[-1]:
            local_err_count += 1
            print(f" *** FAILED *** : mismatch between benchmark and test run: {test_name}")
            print(f"name = {k}\n ref = {v[0]}\n test = {v[1]}\n delta = {v[0] - v[1]}\n")
    if local_err_count == 0:
        print(f"*** PASSED : {test_name} ***")
