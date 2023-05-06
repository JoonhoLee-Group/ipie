import sys
import os
import zmq
import numpy
from ipie.utils.from_trexio import gen_ipie_from_trexio

from mpi4py import MPI
from ipie.analysis.extraction import extract_observable
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric


nsteps = 20
nblocks = 5
timestep = 0.005
port = 5555
trexio_filename = "h2o_dz.h5"
error_threshold = 1.e-3   # Stopping condition
nwalkers = 64


def run_client(seed):

  results = gen_ipie_from_trexio(trexio_filename, verbose=False)

  try: os.mkdir("tmp")
  except: pass
  os.mkdir(f"tmp/{seed}")
  os.chdir(f"tmp/{seed}")

  # The wave function can be built by the server and sent to the client when it connects
  nup = results["nup"]
  ndown = results["ndn"]

  # Obtain MPS trial etc.
  num_elec = nup + ndown
  h1e = results["hcore"]
  chol = results["chol"]
  ecore = results["e0"]
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
  ndets = len(coeff)
#  ndets = 2

  coeff = coeff[:ndets]
  occa = numpy.zeros((ndets, len(occa_list[0])),dtype=numpy.int64)
  occb = numpy.zeros((ndets, len(occb_list[0])),dtype=numpy.int64)

  for i in range(ndets):
      occa[i,:] = occa_list[i]
      occb[i,:] = occb_list[i]


  wavefunction = (coeff, occa, occb)

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

  initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
  walkers = UHFWalkersTrial(trial, initial_walker,
      system.nup, system.ndown, ham.nbasis, nwalkers
  )
  walkers.build(trial)


  # Now let's build our custom AFQMC algorithm

  trial.compute_trial_energy = True
  afqmc_msd = AFQMC(
      comm=MPI.COMM_WORLD,
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

  # Initialization block
  afqmc_msd.run(comm=MPI.COMM_WORLD)



  context = zmq.Context()
  s = context.socket(zmq.REQ)
  s.connect(f"tcp://localhost:{port}")

  running = True
  while running:
    afqmc_msd.run(comm=MPI.COMM_WORLD)
    qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
    data = list(qmc_data["ETotal"][1:])
    s.send_string(str(data))
    running = eval(s.recv())

  print("client closed")

def run_server():
  import socket
  from math import sqrt

  context = zmq.Context()
  s = context.socket(zmq.REP)
  s.bind(f"tcp://*:{port}")
  address = socket.gethostname()
  print(f"tcp://{address}:{port}")

  running = True
  data = []
  while running:
    # Receive the energies
    message = eval(s.recv())
    data += message

    N = len(data)
    average = sum(data)/N
    if N > 2:                  # Compute variance
       l = [ (x-average)*(x-average) for x in data ]
       variance = sum(l)/(N-1.)
    else:
       variance = 0.
    error = sqrt(variance)/sqrt(N)    # Compute error

    print(f"{average} +/- {error}")

    # Stopping condition
    if N > 2 and error < error_threshold:
       running = False

    s.send_string(str(running))

  print("server closed")



if __name__ == "__main__":

  if "-s" in sys.argv:

      run_server()

  elif "-c" in sys.argv:

      run_client(seed=os.getpid())

  else:

      # Spawn clients
      import subprocess
      for _ in range(os.cpu_count()):
         subprocess.Popen(["python", sys.argv[0], "-c"])

      # Run server
      run_server()

