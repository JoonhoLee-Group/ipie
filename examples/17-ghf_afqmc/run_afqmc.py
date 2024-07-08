from pyscf import gto, scf, lo
import numpy, scipy

from ipie.utils.mpi import MPIHandler
from ipie.utils.from_pyscf import generate_integrals
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_uhf_batch,
    local_energy_single_det_ghf_batch
    )
from ipie.hamiltonians.generic import GenericRealChol
from ipie.qmc.afqmc import AFQMC

verbose = True
seed = 7
numpy.random.seed(seed)

# Run PySCF.
mol = gto.M(
        atom=[("H", 0, 0, 0), ("H", (0, 0, 4.2))],
        basis="sto-6g",
        unit="Bohr"
        )
mf_rhf = scf.RHF(mol).run()
psi0a_rhf = mf_rhf.mo_coeff
xinv = numpy.linalg.inv(psi0a_rhf) # work in the RHF basis

mf = scf.UHF(mol).run()
diag_a = [0] * 2
diag_b = [0] * 2
diag_a[0] = 1
diag_b[1] = 1
dm1 = numpy.diag(diag_a), numpy.diag(diag_b)
mf = mf.run(dm1)

psi0a = xinv.dot(mf.mo_coeff[0][:, :mol.nelec[0]])
psi0b = xinv.dot(mf.mo_coeff[1][:, :mol.nelec[1]])

h1e, chol, nuc = generate_integrals(mol, mf.get_hcore(), psi0a_rhf)

num_basis = psi0a_rhf.shape[0]
nchol = chol.shape[0]
nelec = mol.nelec
chol = chol.transpose(1, 2, 0).reshape(num_basis**2, nchol)

# Build Hamiltonian.
ham = GenericRealChol(numpy.array([h1e, h1e]), chol, nuc)

# Build trial wavefunctions.
# UHF.
trial_uhf = SingleDet(numpy.hstack([psi0a, psi0b]), nelec, num_basis, verbose=verbose)
trial_uhf.build()
trial_uhf.half_rotate(ham)

# GHF.
psi0_ghf = numpy.zeros((2*num_basis, numpy.sum(nelec)), dtype=psi0a.dtype)
psi0_ghf[:num_basis, :nelec[0]] = psi0a.copy()
psi0_ghf[num_basis:, nelec[0]:] = psi0b.copy()

# Applying spin-axis rotation. See
# https://en.wikipedia.org/wiki/Eigenspinor#The_spin_1/2_particle
theta = numpy.pi / 2.0
phi = numpy.pi / 4.0

Uspin = numpy.array(
        [[numpy.cos(theta / 2.0), -numpy.exp(1.0j * phi) * numpy.sin(theta / 2.0)],
         [numpy.exp(-1.0j * phi) * numpy.sin(theta / 2.0), numpy.cos(theta / 2.0)]],
        dtype=numpy.complex128)
U = numpy.kron(Uspin, numpy.eye(num_basis))
psi0_ghf = U.dot(psi0_ghf)

trial = SingleDetGHF(psi0_ghf, nelec, num_basis, verbose=verbose)

# Build walkers.
num_walkers = 10
num_steps_per_block = 5
num_blocks = 30
timestep = 0.05

walkers = GHFWalkers(
    psi0_ghf,
    nelec[0],
    nelec[1],
    num_basis,
    num_walkers,
    MPIHandler()
)
walkers.build(trial)

# Build AFQMC driver for GHF trial.
afqmc = AFQMC.build(
            nelec,
            ham,
            trial,
            walkers=walkers,
            num_walkers=num_walkers,
            num_steps_per_block=num_steps_per_block,
            num_blocks=num_blocks,
            timestep=timestep,
            pop_control_freq=5,
            seed=seed,
        )

# The initial energy should be independent of the spin rotation.
trial_uhf.calculate_energy(afqmc.system, afqmc.hamiltonian)
trial.calculate_energy(afqmc.system, afqmc.hamiltonian)
numpy.testing.assert_allclose(trial_uhf.energy, trial.energy)

afqmc.run()
print()

# One can also build GHF trial and walker objects from UHF objects.
# Build SingleDetGHF from SingleDet.
trial = SingleDetGHF(trial_uhf, verbose=verbose)

# Check wavefunctions are identical.
numpy.testing.assert_allclose(trial.psi0[:num_basis, :nelec[0]], trial_uhf.psi0a)
numpy.testing.assert_allclose(trial.psi0[num_basis:, nelec[0]:], trial_uhf.psi0b)

# Build GHFWalkers from UHFWalkers.
walkers_uhf = UHFWalkers(
                numpy.hstack([psi0a, psi0b]),
                nelec[0],
                nelec[1],
                num_basis,
                num_walkers,
                MPIHandler()
                )

walkers = GHFWalkers(walkers_uhf)
walkers.build(trial)

# Build AFQMC driver for GHF trial.
afqmc = AFQMC.build(
            nelec,
            ham,
            trial,
            walkers=walkers,
            num_walkers=num_walkers,
            num_steps_per_block=num_steps_per_block,
            num_blocks=num_blocks,
            timestep=timestep,
            pop_control_freq=5,
            seed=seed,
        )

# Check initial trial and batch energies.
trial_uhf.calculate_energy(afqmc.system, afqmc.hamiltonian)
trial.calculate_energy(afqmc.system, afqmc.hamiltonian)
numpy.testing.assert_allclose(trial_uhf.energy, trial.energy)

energies_uhf = local_energy_single_det_uhf_batch(afqmc.system, ham, walkers_uhf, trial_uhf)
energies = local_energy_single_det_ghf_batch(afqmc.system, ham, walkers, trial)
numpy.testing.assert_allclose(energies_uhf, energies)

afqmc.run()
