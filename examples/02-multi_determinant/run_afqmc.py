"""
Run AFQMC with multi-determinant trial wavefunction.
Both GPU and MPI can be enabled in this script.
"""
import h5py
import numpy
from pyscf import fci, gto, mcscf, scf

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from ipie.walkers.uhf_walkers import UHFWalkersParticleHole
from ipie.utils.mpi import MPIHandler

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    from ipie.qmc.comm import FakeComm
    comm = FakeComm()

from ipie.config import config
# config.update_option("use_gpu", True) # enable GPU
config.update_option("use_gpu", False) # disable GPU

comm_size = comm.size if comm is not None else 1
num_walkers = 640 // comm_size


if comm is None or comm.rank == 0:
    nocca = 4
    noccb = 2

    mol = gto.M(
        atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))],
        basis="ccpvdz",
        verbose=3,
        spin=nocca - noccb,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    ehf = mf.kernel()
    M = 6
    N = 6
    mc = mcscf.CASSCF(mf, M, N)
    mc.chkfile = "scf.chk"
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
    )
    # Need to write wavefunction to checkpoint file.
    with h5py.File("scf.chk", "r+") as fh5:
        fh5["mcscf/ci_coeffs"] = coeff
        fh5["mcscf/occs_alpha"] = occa
        fh5["mcscf/occs_beta"] = occb

    gen_ipie_input_from_pyscf_chk("scf.chk", mcscf=True)

mol_nelec = [8, 6]

with h5py.File("hamiltonian.h5") as fa:
    chol = fa["LXmn"][()]
    h1e = fa["hcore"][()]
    e0 = fa["e0"][()]

num_basis = chol.shape[1]
system = Generic(nelec=mol_nelec)

num_chol = chol.shape[0]
ham = HamGeneric(
    numpy.array([h1e, h1e]),
    chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
    e0,
)

# Build trial wavefunction
with h5py.File("wavefunction.h5", "r") as fh5:
    coeff = fh5["ci_coeffs"][:]
    occa = fh5["occ_alpha"][:]
    occb = fh5["occ_beta"][:]
wavefunction = (coeff, occa, occb)
trial = ParticleHole(
    wavefunction,
    mol_nelec,
    num_basis,
    num_dets_for_props=len(wavefunction[0]),
    verbose=True,
)
trial.compute_trial_energy = True
trial.build()
trial.half_rotate(ham)

initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
numpy.random.seed(123456789)
random_perturbation = numpy.random.random(initial_walker.shape)
initial_walker = initial_walker + random_perturbation
initial_walker, _ = numpy.linalg.qr(initial_walker)
walkers = UHFWalkersParticleHole(
    initial_walker,
    mol_nelec[0],
    mol_nelec[1],
    num_basis,
    num_walkers,
    MPIHandler(),
)
walkers.build(trial)

afqmc_msd = AFQMC.build(
    mol_nelec,
    ham,
    trial,
    walkers=walkers,
    num_walkers=num_walkers,
    num_steps_per_block=25,
    num_blocks=10,
    timestep=0.005,
    stabilize_freq=5,
    seed=96264512,
    pop_control_freq=5,
    verbose=True,
)
afqmc_msd.run()
afqmc_msd.finalise(verbose=True)
