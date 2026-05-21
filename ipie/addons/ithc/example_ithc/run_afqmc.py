# Author: Maxine Luo <man.luo@mpq.mpg.de>
#         Victor Chen <victor.chen@tum.de>
#

"""Run an ithc AFQMC calculation for a H10 chain."""

import numpy as np

np.set_printoptions(suppress=True)
import h5py


from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC
from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from ipie.addons.ithc.qmc.afqmc import AFQMC
from pyscf import gto, scf, ao2mo, fci


def construct_hamil(hamilpath):
    with h5py.File(hamilpath, "r") as f:
        # List all groups
        print("Keys:", list(f.keys()))
        enuc = np.asarray(f["enuc"])
        eri = np.asarray(f["eri"])
        hkin = np.asarray(f["hkin"])
        hnuc = np.asarray(f["hnuc"])
        u = np.asarray(f["u"])
        w = np.asarray(f["w"])

    hamil = GenericITHC(hkin + hnuc, u, w, ecore=enuc)

    return hamil


def construct_trial(hamilpath):
    with h5py.File(hamilpath, "r") as f:
        # List all groups
        print("Keys:", list(f.keys()))
        enuc = np.asarray(f["enuc"])
        eri = np.asarray(f["eri"])
        hkin = np.asarray(f["hkin"])
        hnuc = np.asarray(f["hnuc"])
        nelectron = f["nelectron"][()]

    # print(nelec)
    mol = gto.Mole()
    mol.nelectron = int(nelectron)
    mol.spin = 0
    mol.incore_anyway = True  # no actual integrals from libcint
    mol.build()

    # custom model integrals in orthonormal basis
    norb = hkin.shape[0]
    h1 = hkin + hnuc

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args, **kwargs: h1
    mf.get_ovlp = lambda *args, **kwargs: np.eye(norb)  # orthonormal basis
    mf.energy_nuc = lambda *args, **kwargs: enuc
    mf._eri = ao2mo.restore(8, eri, norb)
    mf.kernel()

    cisolver = fci.FCI(mf)
    efci, fcivec = cisolver.kernel()
    print("\nExact FCI Ground State Energy:", efci)

    # construct trial wavefunction
    na, nb = mol.nelec
    print(mol.nelec)
    psi0a = mf.mo_coeff[:, :na]
    psi0b = mf.mo_coeff[:, :nb]

    trial = SingleDet(wavefunction=np.hstack((psi0a, psi0b)), num_elec=mol.nelec, num_basis=norb)

    trial.build()

    return trial, mol.nelec


hamil = construct_hamil("hamiltonian.h5")
trial, nelec = construct_trial("hamiltonian.h5")  # contains isometry and hamiltonian data
# print(nelec)
# print(f"Shape of trial.psi0a: {np.shape(trial.psi0a)}")
# print(f"Shape of isometry: {np.shape(hamil.isometry)}")
num_walkers = 100
num_steps_per_block = 25
num_blocks = 10  # Adjust this in practice
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
