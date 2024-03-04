import numpy as np
from scipy.optimize import basinhopping

from ipie.legacy.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from ipie.systems.generic import Generic
from ipie.hamiltonians.elph.holstein import HolsteinModel

import jax

def local_energy(
        X: np.ndarray,
        G: np.ndarray,
        m: float,
        w: float,
        g: float,
        nsites: int,
        Lap: np.ndarray,
        T: np.ndarray,
        nup: int,
        ndown: int
    ) -> np.ndarray:

    kinetic_contrib = jax.numpy.einsum('ij->', T[0] * G[0])
    if ndown > 0:
        kinetic_contrib += jax.numpy.einsum('ij->', T[1] * G[1])

    rho = G[0].diagonal() + G[1].diagonal()
    el_ph_contrib = -g * np.sqrt(2 * m * w) * np.sum(rho * X)

    phonon_contrib = m * w**2 * np.sum(X * X) / 2
    phonon_contrib += -0.5 * np.sum(Lap) / m - 0.5 * w * nsites

    local_energy = kinetic_contrib + el_ph_contrib + phonon_contrib
    return local_energy

def gab(A: np.ndarray, B: np.ndarray):
    inv_O = jax.numpy.linalg.inv((A.conj().T).dot(B))                                                                                                                                                            
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB

def get_T(nsites: int, t: float, pbc: bool, ndown: int) -> np.ndarray:
    if nsites == 1:
        return np.array([0.])

    neighbor_map = np.zeros(shape=(nsites,nsites))                      
    for i in range(nsites):
        neighbors = np.array([(i-1)%nsites, (i+1)%nsites])
        neighbor_map[i, neighbors] = 1.
    
    if not pbc:
        neighbor_map[-1,0] = neighbor_map[0,-1] = 0.
    T = -t * neighbor_map
    
    if ndown > 0:
        T = [T.copy(), T.copy()]
    else:
        T = [T,]
    return T

def objective_function(x: np.ndarray, nbasis: int, T: np.ndarray, 
                       g: float, m: float, w0: float, nup: int, ndown: int):
    shift = x[0:nbasis].copy()
    shift = shift.astype(np.float64)
    
    c0a = x[nbasis : (nup+1)*nbasis].copy()
    c0a = jax.numpy.reshape(c0a, (nbasis, nup))
    c0a = c0a.astype(np.float64)
    Ga = gab(c0a, c0a)

    if ndown > 0:
        c0b = x[(nup+1)*nbasis : ].copy()
        c0b = jax.numpy.reshape(c0b, (nbasis, ndown))
        c0b = c0b.astype(np.float64)
        Gb = gab(c0b, c0b)
    else:
        Gb = jax.numpy.zeros_like(Ga, dtype=np.float64)
    
    G = [Ga, Gb]

    phi = HarmonicOscillator(m, w0, order=0, shift=shift)
    Lap = phi.laplacian(shift)

    etot = local_energy(shift, G, m, w0, g, nbasis, Lap, T, nup, ndown)
    return etot.real


def gradient(x: np.ndarray, nbasis: int, T: np.ndarray, 
             g: float, m: float, w0: float, nup: int, ndown: int):
    grad = np.array(
        jax.grad(objective_function)(
            x, nbasis, T, g, m, w0, nup, ndown
        ),
        dtype=np.float64
    )
    return grad

def func(x: np.ndarray, nbasis: int, T: np.ndarray, 
         g: float, m: float, w0: float, nup: int, ndown: int):
    f = objective_function(x, nbasis, T, g, m, w0, nup, ndown)
    df = gradient(x, nbasis, T, g, m, w0, nup, ndown)
    return f, df

def print_fun(x: np.ndarray, f: float, accepted: bool):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))

def variational_trial(init_phonons: np.ndarray, init_electron: np.ndarray, hamiltonian, system):

    init_phonons = init_phonons.astype(np.float64)
    init_electron = init_electron.astype(np.float64).ravel()

    x = np.hstack([init_phonons, init_electron])

    #TODO this could definitely be hamiltonian.T
#    T = get_T(hamiltonian.nsites, hamiltonian.t, hamiltonian.pbc, system.ndown)
    T = hamiltonian.T

    maxiter = 500
    minimizer_kwargs = {
        "jac": True,
        "args": (hamiltonian.nsites, T, hamiltonian.g, hamiltonian.m, hamiltonian.w0, system.nup, system.ndown),
        "options": {
            "gtol": 1e-10,
            "eps": 1e-10,
            "maxiter": maxiter,
            "disp": False,
        },
    }

    res = basinhopping(
        func,
        x,
        minimizer_kwargs=minimizer_kwargs,
        callback=print_fun,
        niter=maxiter,
        niter_success=3,
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

if __name__ == '__main__':
    
    nup = 1
    ndown = 1
    system = Generic([nup, ndown])

    g = 1.
    t = 1.
    w0 = 1.
    nsites = 3
    pbc = True

    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)

    init_phonons = np.array([0.1, 0.1, 0.1], dtype=np.float64)
#    init_electron = np.array([1.,1.], dtype=np.float64)
#    init_electron /= np.linalg.norm(init_electron)
    init_electron = np.array([[1,0,0],[0,1,0]])

    print(variational_trial(init_phonons, init_electron, ham, system))

