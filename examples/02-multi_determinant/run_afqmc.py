import numpy
from mpi4py import MPI
from ipie.analysis.extraction import extract_observable
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
import h5py


comm = MPI.COMM_WORLD

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
    e0
)

# Build Trial

# 4. Build walkers
with h5py.File("wavefunction.h5", 'r') as fh5:
    coeff = fh5['ci_coeffs'][:]
    occa = fh5['occ_alpha'][:]
    occb = fh5["occ_beta"][:]

# 3. Build trial wavefunction
wavefunction = (coeff, occa, occb)
from ipie.trial_wavefunction.particle_hole import ParticleHoleNonChunked
trial = ParticleHoleNonChunked(
    wavefunction,
    mol_nelec,
    num_basis,
    num_dets_for_props=len(wavefunction[0])
)
trial.compute_trial_energy = False
trial.build()
trial.half_rotate(ham)


afqmc_msd = AFQMC.build(
    mol_nelec,
    ham,
    trial,
    num_walkers=10,
    num_steps_per_block=25,
    num_blocks=10,
    timestep=0.005,
    stabilize_freq=5,
    pop_control_freq=5
)
afqmc_msd.run()
afqmc_msd.finalise(verbose=True)

qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
y2 = qmc_data["ETotal"]
y2 = y2[1:]

