import numpy
from mpi4py import MPI

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.utils.from_trexio import gen_ipie_from_trexio

trexio_filename = "h2o_dz.h5"

comm = MPI.COMM_WORLD

num_frozen_core = 1

if comm.rank == 0:
    print("Hartree-Fock energy: -76.0267720534593")
    print("CI energy          : -76.1665620477625")
    results = gen_ipie_from_trexio(trexio_filename, num_frozen_core=num_frozen_core)
    nup = results["nup"]
    ndown = results["ndn"]

    h1e = results["hcore"]
    chol = results["chol"]
    ecore = results["e0"]
    coeff = results["ci_coeffs"]
    occa_list = results["occa"]
    occb_list = results["occb"]

else:
    nup = None
    ndown = None
    h1e = None
    chol = None
    ecore = None
    coeff = None
    occa_list = None
    occb_list = None

nup = comm.bcast(nup, root=0)
ndown = comm.bcast(ndown, root=0)
h1e = comm.bcast(h1e, root=0)
chol = comm.bcast(chol, root=0)
ecore = comm.bcast(ecore, root=0)
coeff = comm.bcast(coeff, root=0)
occa_list = comm.bcast(occa_list, root=0)
occb_list = comm.bcast(occb_list, root=0)

num_elec = nup + ndown
nbasis = h1e.shape[0]
nchol = chol.shape[-1]
mo_coeff = numpy.eye(nbasis)

# Build System
system = Generic(nelec=(nup, ndown))
# Build Hamiltonian
ham = HamGeneric(
    numpy.array([h1e, h1e]),
    chol.reshape((nbasis * nbasis, nchol)),
    ecore,
)

# Build Trial

# 4. Build walkers

nocca, noccb = nup, ndown
nelec = (nocca, noccb)
system = Generic(nelec=nelec)

# 3. Build trial wavefunction
ndets = len(coeff)
coeff = coeff[:ndets]
occa = numpy.zeros((ndets, len(occa_list[0])), dtype=numpy.int64)
occb = numpy.zeros((ndets, len(occb_list[0])), dtype=numpy.int64)

for i in range(ndets):
    occa[i, :] = occa_list[i]
    occb[i, :] = occb_list[i]

wavefunction = (coeff, occa, occb)

from ipie.trial_wavefunction.particle_hole import ParticleHoleWicks

trial = ParticleHoleWicks(
    wavefunction, (nocca, noccb), nbasis, num_dets_for_props=len(wavefunction[0])
)
trial.build()
trial.half_rotate(ham, comm=comm)

# 4. Build walkers
from ipie.walkers.walkers_dispatch import UHFWalkersTrial

nwalkers = 640 // comm.size
initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
walkers = UHFWalkersTrial(trial, initial_walker, system.nup, system.ndown, ham.nbasis, nwalkers)
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
    seed=seed,
    verbose=1,
)
afqmc_msd.run(comm=comm)
afqmc_msd.finalise(verbose=True)
