"""Example on how to use secant method to find optimal chemical potential.

A final iteration at the optimal chemical potential is performed with the
results stored in optimal.0.h5.

Usage: srun -n 144 -N 4 python find_chem_pot.py
"""
import numpy
from mpi4py import MPI
import sys
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
from pauxy.systems.ueg import UEG
from pauxy.qmc.options import QMCOpts
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.analysis.thermal import analyse_energy


def determine_nav(comm, options, mu, target):
    options['system']['mu'] = mu
    afqmc = ThermalAFQMC(comm, options=options,
                         verbose=(comm.rank==0))
    afqmc.run(comm=comm, verbose=True)
    if comm.rank == 0:
        av = analyse_energy([afqmc.estimators.h5f_name])
        nav = av.Nav.values[0]
    else:
        nav = None
    nav = comm.bcast(nav, root=0)
    return nav - target

def secant(comm, options, x0, x1, target, maxiter=10, threshold=1e-3):
    nx0 = determine_nav(comm, options, x0, target)
    for it in range(0, maxiter):
        nx1 = determine_nav(comm, options, x1, target)
        xn = x1 - nx1 * (x1-x0) / (nx1-nx0)
        if comm.rank == 0:
            print(" # Chemical potential iteration: {} {} {} {} {} {}"
                  "".format(it, x0, x1, nx0, nx1, xn))
        if abs(xn-x1) < threshold:
            break
        x0 = x1
        x1 = xn
        nx0 = nx1
    return xn

def find_mu_opt(options):
    comm = MPI.COMM_WORLD
    # Guess initial chemical potential from trial density matrix (mu_xc < 0)
    system = UEG(options['system'])
    qmcopt = QMCOpts(options['qmc'], system)
    trial = OneBody(comm, system, qmcopt.beta, qmcopt.dt)
    mu0 = trial.mu
    # guess for bracket.
    mu1 = mu0 - 0.5*abs(mu0)
    mu_opt = secant(comm, options, mu0, mu1, system.ne)
    if comm.rank == 0:
        print("# Converged mu: {}".format(mu_opt))
    # Run longer simulation at optimal mu.
    sys_opts['mu'] = mu_opt
    qmc['nsteps'] = 50
    estim['basename'] = 'optimal'
    afqmc = ThermalAFQMC(comm, options=options,
                         verbose=(comm.rank==0))
    afqmc.run(comm=comm, verbose=True)

if __name__ == '__main__':
    sys_opts = {
        "name": "UEG",
        "nup": 33,
        "ndown": 33,
        "rs": 2.0,
        "ecut": 2.5
    }
    # Relatively coarse sampling for root finding.
    qmc = {
        "dt": 0.05,
        "nsteps": 20,
        "nwalkers": 288,
        "scaled_temperature": True,
        "beta": 1.0
    }
    # Ensure different iterations do not write to the same file.
    estim = {"overwrite": False}
    options = {"qmc": qmc, "estimators": estim, "system": sys_opts}
    find_mu_opt(options)
