# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.optimize import basinhopping

from ipie.addons.eph.trial_wavefunction.variational.estimators import gab

import jax


def local_energy(
    X: np.ndarray,
    G: np.ndarray,
    m: float,
    w: float,
    g: float,
    nsites: int,
    T: np.ndarray,
    nup: int,
    ndown: int,
) -> np.ndarray:

    kinetic = np.sum(T[0] * G[0] + T[1] * G[1])
    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = -g * 2 * np.sqrt(m * w) * np.sum(rho * X)
    e_ph = m * w**2 * np.sum(X * X)

    local_energy = kinetic + e_eph + e_ph
    return local_energy


def objective_function(
    x: np.ndarray, nbasis: int, T: np.ndarray, g: float, m: float, w0: float, nup: int, ndown: int
):
    shift = x[0:nbasis].copy()
    shift = shift.astype(np.float64)

    c0a = x[nbasis : (nup + 1) * nbasis].copy()
    c0a = jax.numpy.reshape(c0a, (nbasis, nup))
    c0a = c0a.astype(np.float64)
    Ga = gab(c0a, c0a)

    if ndown > 0:
        c0b = x[(nup + 1) * nbasis :].copy()
        c0b = jax.numpy.reshape(c0b, (nbasis, ndown))
        c0b = c0b.astype(np.float64)
        Gb = gab(c0b, c0b)
    else:
        Gb = jax.numpy.zeros_like(Ga, dtype=np.float64)

    G = [Ga, Gb]

    etot = local_energy(shift, G, m, w0, g, nbasis, T, nup, ndown)

    return etot.real


def gradient(
    x: np.ndarray, nbasis: int, T: np.ndarray, g: float, m: float, w0: float, nup: int, ndown: int
):
    grad = np.array(
        jax.grad(objective_function)(x, nbasis, T, g, m, w0, nup, ndown), dtype=np.float64
    )
    return grad


def func(
    x: np.ndarray, nbasis: int, T: np.ndarray, g: float, m: float, w0: float, nup: int, ndown: int
):
    f = objective_function(x, nbasis, T, g, m, w0, nup, ndown)
    df = gradient(x, nbasis, T, g, m, w0, nup, ndown)
    return f, df


def print_fun(x: np.ndarray, f: float, accepted: bool):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


def variational_trial(init_phonons: np.ndarray, init_electron: np.ndarray, hamiltonian, system):

    init_phonons = init_phonons.astype(np.float64)
    init_electron = init_electron.astype(np.float64).ravel()

    x = np.hstack([init_phonons, init_electron])

    maxiter = 500
    minimizer_kwargs = {
        "jac": True,
        "args": (
            hamiltonian.nsites,
            hamiltonian.T,
            hamiltonian.g,
            hamiltonian.m,
            hamiltonian.w0,
            system.nup,
            system.ndown,
        ),
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

    beta_shift = res.x[: hamiltonian.nsites]
    if system.ndown > 0:
        psia = res.x[hamiltonian.nsites : hamiltonian.nsites * (system.nup + 1)]
        psia = psia.reshape((hamiltonian.nsites, system.nup))
        psib = res.x[hamiltonian.nsites * (system.nup + 1) :]
        psib = psib.reshape((hamiltonian.nsites, system.ndown))
        psi = np.column_stack([psia, psib])
    else:
        psia = res.x[hamiltonian.nsites :].reshape((hamiltonian.nsites, system.nup))
        psi = psia

    return etrial, beta_shift, psi
