import matplotlib.pyplot as pl
import numpy
import pandas as pd

from ipie.config import MPI
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.estimators.thermal import particle_number
from ipie.legacy.qmc.thermal_afqmc import ThermalAFQMC
from ipie.legacy.trial_density_matrices.mean_field import MeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody
from ipie.systems.generic import Generic

comm = MPI.COMM_WORLD

sys_opts = {
    "name": "Generic",
    "nup": 2,
    "ndown": 2,
    "mu": 0.2,
    "sparse": False,
    "integrals": "hamil.h5",
}

system = Generic(inputs=sys_opts)

# trial = OneBody(comm, system, 1.0, 0.05, verbose=True)
mus = numpy.arange(-1, 1)
data = []
dt = 0.05
fci = pd.read_csv("be_fixed_n.out", sep=r"\s+")
for b, n in zip(fci.beta, fci.N):
    trial = OneBody(comm, system, b, dt, options={"nav": n}, verbose=True)
    data.append(
        [
            local_energy(system, trial.P, opt=False)[0].real,
            particle_number(trial.P).real,
        ]
    )
pl.plot(fci.beta, zip(*data)[0], label="Match N")
match = zip(*data)[0]
data = []
for b, n in zip(fci.beta, fci.N):
    trial = MeanField(comm, system, b, dt, options={"nav": n}, verbose=True)
    data.append(
        [
            local_energy(system, trial.P, opt=False)[0].real,
            particle_number(trial.P).real,
        ]
    )
pl.plot(fci.beta, fci.E, label="FCI")
pl.plot(fci.beta, zip(*data)[0], label="THF", linestyle=":")
data = pd.DataFrame({"beta": fci.beta, "FCI": fci.E, "THF": zip(*data)[0], "Match": match})
print(data.to_string())
# trial = MeanField(comm, system, beta, dt,
# options={"nav": fci.N.values[-1]}, verbose=True)
# trial.thermal_hartree_fock(system, beta)
# print(trial.
data = []
# for m in fci.mu:
# trial = OneBody(comm, system, beta, dt, options={"mu": m})
# data.append([trial.calculate_energy(system, beta)[0].real,
# trial.calculate_nav(system, beta).real])
# pl.plot(fci.mu, zip(*data)[0], label='Same mu')
pl.legend()
pl.show()
