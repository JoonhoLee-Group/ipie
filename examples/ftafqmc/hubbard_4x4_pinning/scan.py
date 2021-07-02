from mpi4py import MPI
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
import numpy

comm = MPI.COMM_WORLD

system = {
    "name": "Hubbard",
    "nup": 7,
    "ndown": 7,
    "symmetric": True,
    "pinning_fields": True,
    "nx": 4,
    "ny": 4,
    "U": 4,
    "mu": 1
}

qmc = {
    "dt": 0.05,
    "nsteps": 10,
    "nwalkers": 288,
    "beta": 5
}

mu = numpy.linspace(0.8,1,5)
for i, b in enumerate([2, 5, 10, 20]):
    estim = {
        "overwrite": False,
        "basename": "estimates_beta_{}".format(i)
        }
    qmc["beta"] = b
    # Scan over chemical potential values
    for m in mu:
        system['mu'] = m
        afqmc = ThermalAFQMC(comm, system, qmc, estimates=estim,
                             verbose=(True and comm.rank==0))
        afqmc.run(comm=comm, verbose=True)

# Rerun with optimised chemical potential values.
# These are values from reference.
# mus = numpy.array([0.44086182875227903,0.4611368640132099,
                   # 0.4769148405517905,0.4863196188379373])
# mus = 2.0 * mus
for i, b in enumerate(beta):
    files = glob.glob('estimates_beta_'+str(i)+'*.h5')
    if comm.rank == 0:
        data = analyse_energy(files)
        vol = get_sys_param(files[0], 'vol')
        mu = find_chem_pot(data, 0.875, vol)
        print(b, mu, vol)
    else:
        mu = None
    mu = comm.bcast(mu, root=0)
    system["mu"] = mu
    qmc["beta"] = b
    estim = {
        "overwrite": False,
        "basename": "final_av_{}".format(i),
        "mixed": {
            "average_gf": True
        }
    }
    afqmc = ThermalAFQMC(comm, system, qmc, estimates=estim,
                         verbose=(True and comm.rank==0))
    afqmc.run(comm=comm, verbose=True)
