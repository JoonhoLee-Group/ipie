from pyscf import gto, scf, lo
import numpy, scipy
from ipie.utils.from_pyscf import generate_integrals
from ipie.qmc.afqmc import AFQMC

mol = gto.M(
    atom=[("H", 0, 0, 0), ("H", (0, 0, 4.2))],
    basis="sto-6g",
    unit="Bohr"
)
mf_rhf = scf.RHF(mol).run()
phi0a_rhf = mf_rhf.mo_coeff
xinv = numpy.linalg.inv(phi0a_rhf) # work in the RHF basis

mf = scf.UHF(mol).run()
diag_a = [0] * 2
diag_b = [0] * 2
diag_a[0] = 1
diag_b[1] = 1
dm1 = numpy.diag(diag_a), numpy.diag(diag_b)
mf = mf.run(dm1)

phi0a = xinv.dot(mf.mo_coeff[0][:, :mol.nelec[0]])
phi0b = xinv.dot(mf.mo_coeff[1][:, :mol.nelec[1]])

h1e, chol, nuc = generate_integrals(
    mol,
    mf.get_hcore(),
    phi0a_rhf,
)

num_basis = phi0a_rhf.shape[0]
nchol = chol.shape[0]
nelec = mol.nelec
chol = chol.transpose(1, 2, 0).reshape(num_basis * num_basis, nchol)

from ipie.hamiltonians.generic import GenericRealChol
ham = GenericRealChol(numpy.array([h1e, h1e]), chol, nuc)

from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.trial_wavefunction.single_det import SingleDet
psi0_ghf = numpy.zeros((2 * num_basis, nelec[0] + nelec[1]), dtype=phi0a.dtype)
psi0_ghf[:num_basis, :nelec[0]] = phi0a.copy()
psi0_ghf[num_basis:, nelec[0]:] = phi0b.copy()

theta = numpy.pi / 4.0

Uspin = numpy.array(
    [
        [numpy.cos(theta / 2.0), numpy.sin(theta / 2.0)],
        [-numpy.sin(theta / 2.0), numpy.cos(theta / 2.0)],
    ],
    dtype=numpy.complex128,
)
U = numpy.kron(Uspin, numpy.eye(num_basis))
psi0 = U.dot(psi0_ghf)

trial = SingleDetGHF(
    psi0_ghf,
    nelec,
    num_basis,
)


trial_uhf = SingleDet(
    numpy.hstack([phi0a, phi0b]),
    nelec,
    num_basis
)
trial_uhf.half_rotate(ham)

num_walkers = 10
num_steps_per_block = 25
num_blocks = 100
timestep = 0.005


from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.utils.mpi import MPIHandler
walkers = UHFWalkers(
    numpy.hstack([phi0a, phi0a]),
    nelec[0],
    nelec[1],
    num_basis,
    num_walkers,
    MPIHandler()
)
walkers.build(trial_uhf)

from ipie.walkers.ghf_walkers import GHFWalkers
ghf_walkers = GHFWalkers(walkers)
ghf_walkers.build(trial)

afqmc = AFQMC.build(
    nelec,
    ham,
    trial,
    walkers=ghf_walkers,
    num_walkers=num_walkers,
    num_steps_per_block=num_steps_per_block,
    num_blocks=num_blocks,
    timestep=timestep,
    pop_control_freq=5,
    seed=59306159,
)
afqmc.run()