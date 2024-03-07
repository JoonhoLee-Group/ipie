import numpy as np
from scipy.optimize import minimize, basinhopping 
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab

import jax
import jax.numpy as npj

def gradient_toyozawa_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted):
    grad = np.array(jax.grad(objective_function_toyozawa_mo)(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted))
    return grad

def objective_function_toyozawa_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted, zero_th=1e-12):
    nbasis = int(round(nbasis))
    nup = int(round(nup))
    ndown = int(round(ndown))
    
    shift0 = x[:nbasis]

    nbsf = nbasis
    nocca = nup
    noccb = ndown
    nvira = nbasis - nocca
    nvirb = nbasis - noccb
    
    psi0a = npj.array(x[nbsf:nbsf+nbsf*nocca],dtype=npj.float64)
    psi0a = psi0a.reshape((nocca, nbsf)).T

    if (noccb>0):
        psi0b = npj.array(x[nbsf+nbsf*nocca : ], dtype=npj.float64)
        psi0b = psi0b.reshape((noccb, nbsf)).T
    
    nperms = len(perms)

    num_energy = 0.
    denom = 0.0
        
    beta0 = shift0 * npj.sqrt(m * w0 /2.0)

    for i,perm in enumerate(perms):    
        psia_j = psi0a[perm, :]
        beta_j = beta0[npj.array(perm)]
        
        if (noccb > 0):
            psib_j = psi0b[perm, :]
            overlap = npj.linalg.det(psi0a.T.dot(psia_j)) * npj.linalg.det(psi0b.T.dot(psib_j)) * npj.prod(npj.exp(-0.5 * (beta0**2 + beta_j**2) + beta0*beta_j))
        else:
            overlap = npj.linalg.det(psi0a.T.dot(psia_j)) * npj.prod(npj.exp (- 0.5 * (beta0**2 + beta_j**2) + beta0*beta_j))

        if overlap < zero_th:
            continue

        Ga_j = gab(psi0a, psia_j)
        if (noccb > 0):
            Gb_j = gab(psi0b, psib_j)
        else:
            Gb_j = npj.zeros_like(Ga_j)

        G_j = [Ga_j, Gb_j]

        rho = G_j[0].diagonal() + G_j[1].diagonal()
        ke = npj.sum(T[0] * G_j[0] + T[1] * G_j[1])
        pe = U * npj.dot(G_j[0].diagonal(), G_j[1].diagonal())
        e_ph = w0 * npj.sum(beta0 * beta_j) 
        e_eph = - g * npj.dot(rho, beta0 + beta_j)

        num_energy += (ke + pe + e_ph + e_eph) * overlap  # 2.0 comes from hermiticity
        denom += overlap

    etot = num_energy / denom
    return etot.real

def circ_perm(lst):
    cpy = lst[:]
    yield cpy
    for i in range(len(lst) - 1):
        cpy = cpy[1:] + [cpy[0]]
        yield cpy

def func(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted):
    f = objective_function_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted)
    df = gradient_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted)
    return f, df

def print_fun(x: np.ndarray, f: float, accepted: bool):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))

def func_toyo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted):
    f = objective_function_toyozawa_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted)
    df = gradient_toyozawa_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted)
    return f, df

def variational_trial_toyozawa(shift_init: np.ndarray, electron_init: np.ndarray, hamiltonian, system):
    psi = electron_init.T.real.ravel()
    shift = shift_init.real

    perms = list(circ_perm([i for i in range(hamiltonian.nsites)]))

    x = np.zeros((system.nup + system.ndown + 1) * hamiltonian.nsites)
    x[:hamiltonian.nsites] = shift.copy()
    x[hamiltonian.nsites:] = psi.copy() #[:,0]

    res = minimize(
            objective_function_toyozawa_mo, 
            x, 
            args = (float(hamiltonian.nsites), float(system.nup), float(system.ndown), 
                    hamiltonian.T, 0., hamiltonian.g, hamiltonian.m, hamiltonian.w0, perms, False), 
            jac = gradient_toyozawa_mo, 
            tol = 1e-10,
            method='L-BFGS-B', 
            options = {
                'maxls': 20, 
                'iprint': 2, 
                'gtol': 1e-10, 
                'eps': 1e-10, 
                'maxiter': 15000,
                'ftol': 1.0e-10, 
                'maxcor': 1000, 
                'maxfun': 15000,
                'disp':True
            }
    )

    etrial = res.fun
    beta_shift = res.x[:hamiltonian.nsites]
    if system.ndown > 0:
        psia = res.x[hamiltonian.nsites : hamiltonian.nsites*(system.nup+1)]
        psia = psia.reshape((hamiltonian.nsites, system.nup))
        psib = res.x[hamiltonian.nsites*(system.nup+1) : ]
        psib = psib.reshape((hamiltonian.nsites, system.ndown))
        psi = np.column_stack([psia, psib])
    else:
        psia = res.x[hamiltonian.nsites:].reshape((hamiltonian.nsites, system.nup))
        psi = psia
 

    return etrial, beta_shift, psi
