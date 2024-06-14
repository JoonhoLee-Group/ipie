import torch
import h5py
from ipie.addons.adafqmc.utils.miscellaneous import read_input, generate_hamiltonianobs_shared, trial_tangent
from mpi4py import MPI
from ipie.addons.adafqmc.qmc.adafqmc import ADAFQMC
import argparse
from ipie.utils.mpi import get_shared_comm

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# read hamiltonian, trial and tangent
if rank == 0:
    with h5py.File('trial_with_tangent.h5', 'r') as f:
        trial = torch.from_numpy(f['trial'][()])
        tg = torch.from_numpy(f['tangent'][()])
else:
    trial = None
    tg = None

scomm = get_shared_comm(comm, verbose=True)
hamobs = generate_hamiltonianobs_shared(scomm, 'my_hamiltonian.h5', 'dipole')
trial = comm.bcast(trial, root=0)
tg = comm.bcast(tg, root=0)

#print(f"trial = {trial_detached} on rank {rank}")

options = read_input(args.file)
if rank == 0:
    print(options)

adafqmc = ADAFQMC.build(comm, trial_tangent, **options)
if rank == 0:
    print("afqmc prepared")
energy, obs = adafqmc.run(hamobs, trial, tg)
