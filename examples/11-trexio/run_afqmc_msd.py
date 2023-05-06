import numpy
from ipie.utils.from_trexio import gen_ipie_from_trexio

from mpi4py import MPI
from ipie.analysis.extraction import extract_observable
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric


print("Hartree-Fock energy: -76.0267720534593")
print("CI energy          : -76.1665620477625")
#trexio_filename = "h2o_dz.h5"
trexio_filename = "h2o_dz.trexio"

results = gen_ipie_from_trexio(trexio_filename)

nup = results["nup"]
ndown = results["ndn"]


comm = MPI.COMM_WORLD

# Obtain MPS trial etc.
num_elec = nup + ndown
h1e = results["hcore"]
chol = results["chol"]
ecore = results["e0"]
ecore = 9.18953375861449
mo_coeff = numpy.eye(h1e.shape[0])
nbasis = h1e.shape[0]
nchol = chol.shape[-1]

# Build System
system = Generic(nelec=(nup,ndown))

# Build Hamiltonian
ham = HamGeneric(
    numpy.array([h1e, h1e]),
    chol.reshape((nbasis * nbasis, nchol)),
    ecore,
)

# Build Trial

# 4. Build walkers
coeff = results["ci_coeffs"]
occa_list = results["occa"]
occb_list = results["occb"]

nocca, noccb = nup, ndown
nelec = (nocca,noccb)
system = Generic(nelec=nelec)

# 3. Build trial wavefunction
# ndets = len(coeff)
ndets = 10

coeff = coeff[:ndets]
occa = numpy.zeros((ndets, len(occa_list[0])),dtype=numpy.int64)
occb = numpy.zeros((ndets, len(occb_list[0])),dtype=numpy.int64)

for i in range(ndets):
    occa[i,:] = occa_list[i]
    occb[i,:] = occb_list[i]


wavefunction = (coeff, occa, occb)

# coeff = coeff[:2]
# occa = occa[:2]
# occb = occb[:2]
#coeff = [coeff[0]]
#occa = [occa[0]]
#occb = [occb[0]]
#wavefunction = (coeff, occa, occb)

from ipie.trial_wavefunction.particle_hole import ParticleHoleWicks
trial = ParticleHoleWicks(
    wavefunction,
    (nocca, noccb),
    nbasis,
    num_dets_for_props=len(wavefunction[0])
)
trial.build()
trial.half_rotate(ham)

# 4. Build walkers
from ipie.walkers.walkers_dispatch import UHFWalkersTrial

nwalkers = 640

initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
walkers = UHFWalkersTrial(trial, initial_walker,
    system.nup, system.ndown, ham.nbasis, nwalkers
)
walkers.build(trial)


# Now let's build our custom AFQMC algorithm

nsteps = 25
nblocks = 100
timestep = 0.005
seed = 7

trial.compute_trial_energy = True
afqmc_msd = AFQMC(
    comm,
    system=system,
    hamiltonian=ham,
    trial=trial,
    walkers=walkers,
    nwalkers=nwalkers,
    num_steps_per_block=nsteps,
    num_blocks=nblocks,
    timestep=timestep,
    seed = seed,
)
afqmc_msd.run(comm=comm)
afqmc_msd.finalise(verbose=True)

qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
y2 = qmc_data["ETotal"]
y2 = y2[1:]

