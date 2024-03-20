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
from scipy.optimize import minimize
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab

import jax
import jax.numpy as npj


def gradient_toyozawa_mo(x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted):
    grad = np.array(
        jax.grad(objective_function_toyozawa_mo)(
            x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted
        )
    )
    return grad


def objective_function_toyozawa_mo(
    x, nbasis, nup, ndown, T, U, g, m, w0, perms, restricted, zero_th=1e-12
):
    nbasis = int(round(nbasis))
    nup = int(round(nup))
    ndown = int(round(ndown))

    shift0 = x[:nbasis]

    nbsf = nbasis
    nocca = nup
    noccb = ndown

    psi0a = npj.array(x[nbsf : nbsf + nbsf * nocca], dtype=npj.float64)
    psi0a = psi0a.reshape((nocca, nbsf)).T

    if noccb > 0:
        psi0b = npj.array(x[nbsf + nbsf * nocca :], dtype=npj.float64)
        psi0b = psi0b.reshape((noccb, nbsf)).T

    num_energy = 0.0
    denom = 0.0

    beta0 = shift0 * npj.sqrt(m * w0 / 2.0)

    for perm in perms:
        psia_j = psi0a[perm, :]
        beta_j = beta0[npj.array(perm)]

        if noccb > 0:
            psib_j = psi0b[perm, :]
            overlap = (
                npj.linalg.det(psi0a.T.dot(psia_j))
                * npj.linalg.det(psi0b.T.dot(psib_j))
                * npj.prod(npj.exp(-0.5 * (beta0**2 + beta_j**2) + beta0 * beta_j))
            )
        else:
            overlap = npj.linalg.det(psi0a.T.dot(psia_j)) * npj.prod(
                npj.exp(-0.5 * (beta0**2 + beta_j**2) + beta0 * beta_j)
            )

        if overlap < zero_th:
            continue

        Ga_j = gab(psi0a, psia_j)
        if noccb > 0:
            Gb_j = gab(psi0b, psib_j)
        else:
            Gb_j = npj.zeros_like(Ga_j)

        G_j = [Ga_j, Gb_j]

        rho = G_j[0].diagonal() + G_j[1].diagonal()
        ke = npj.sum(T[0] * G_j[0] + T[1] * G_j[1])
        pe = U * npj.dot(G_j[0].diagonal(), G_j[1].diagonal())
        e_ph = w0 * npj.sum(beta0 * beta_j)
        e_eph = -g * npj.dot(rho, beta0 + beta_j)

        num_energy += (ke + pe + e_ph + e_eph) * overlap  # 2.0 comes from hermiticity
        denom += overlap

    etot = num_energy / denom
    return etot.real


def circ_perm(lst: np.ndarray) -> np.ndarray:
    """Returns a matrix which rows consist of all possible
    cyclic permutations given an initial array lst.

    Parameters
    ----------
    lst :
        Initial array which is to be cyclically permuted
    """
    circs = lst
    for shift in range(1, len(lst)):
        new_circ = np.roll(lst, -shift)
        circs = np.vstack([circs, new_circ])
    return circs


def variational_trial_toyozawa(
    shift_init: np.ndarray, electron_init: np.ndarray, hamiltonian, system, verbose=2
):
    psi = electron_init.T.real.ravel()
    shift = shift_init.real

    perms = circ_perm(np.arange(hamiltonian.nsites))

    x = np.zeros((system.nup + system.ndown + 1) * hamiltonian.nsites)
    x[: hamiltonian.nsites] = shift.copy()
    x[hamiltonian.nsites :] = psi.copy()

    res = minimize(
        objective_function_toyozawa_mo,
        x,
        args=(
            float(hamiltonian.nsites),
            float(system.nup),
            float(system.ndown),
            hamiltonian.T,
            0.0,
            hamiltonian.g,
            hamiltonian.m,
            hamiltonian.w0,
            perms,
            False,
        ),
        jac=gradient_toyozawa_mo,
        tol=1e-10,
        method="L-BFGS-B",
        options={
            "maxls": 20,
            "iprint": verbose,
            "gtol": 1e-10,
            "eps": 1e-10,
            "maxiter": 15000,
            "ftol": 1.0e-10,
            "maxcor": 1000,
            "maxfun": 15000,
            "disp": True,
        },
    )

    etrial = res.fun
    beta_shift = res.x[: hamiltonian.nsites]
    if system.ndown > 0:
        psia = res.x[hamiltonian.nsites : hamiltonian.nsites * (system.nup + 1)]
        psia = psia.reshape((system.nup, hamiltonian.nsites)).T
        psib = res.x[hamiltonian.nsites * (system.nup + 1) :]
        psib = psib.reshape((system.ndown, hamiltonian.nsites)).T
        psi = np.column_stack([psia, psib])
    else:
        psia = res.x[hamiltonian.nsites :].reshape((system.nup, hamiltonian.nsites)).T
        psi = psia

    return etrial, beta_shift, psi
