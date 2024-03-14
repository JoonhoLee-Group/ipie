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
#
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

from typing import Tuple, Union

import numpy

from ipie.addons.free_projection.propagation.free_propagation import FreePropagation
from ipie.addons.free_projection.qmc.fp_afqmc import FPAFQMC
from ipie.addons.free_projection.walkers.uhf_walkers import UHFWalkersFP
from ipie.hamiltonians import Generic as HamGeneric
from ipie.qmc.options import QMCOpts
from ipie.systems import Generic
from ipie.utils.io import get_input_value
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import build_random_trial, generate_hamiltonian, TestData


def build_test_case_handlers_fp(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    trial_type="single_det",
    wfn_type="opt",
    complex_integrals: bool = False,
    complex_trial: bool = False,
    seed: Union[int, None] = None,
    rhf_trial: bool = False,
    two_body_only: bool = False,
    choltol: float = 1e-3,
    reortho: bool = True,
    options: Union[dict, None] = None,
):
    if seed is not None:
        numpy.random.seed(seed)
    sym = 8
    if complex_integrals:
        sym = 4
    h1e, chol, _, eri = generate_hamiltonian(
        num_basis, num_elec, cplx=complex_integrals, sym=sym, tol=choltol
    )
    system = Generic(nelec=num_elec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, num_basis**2)).T.copy(),
        ecore=0,
    )
    ham.eri = eri.copy()
    trial, init = build_random_trial(
        num_elec,
        num_basis,
        num_dets=num_dets,
        wfn_type=wfn_type,
        trial_type=trial_type,
        complex_trial=complex_trial,
        rhf_trial=rhf_trial,
    )
    trial.half_rotate(ham)
    trial.calculate_energy(system, ham)
    # necessary for backwards compatabilty with tests
    if seed is not None:
        numpy.random.seed(seed)

    nwalkers = get_input_value(options, "nwalkers", default=10, alias=["num_walkers"])
    walkers = UHFWalkersFP(init, system.nup, system.ndown, ham.nbasis, nwalkers, MPIHandler())
    walkers.build(trial)  # any intermediates that require information from trial

    prop = FreePropagation(time_step=options["dt"])
    prop.build(ham, trial)

    trial.calc_greens_function(walkers)
    for _ in range(options.num_steps):
        if two_body_only:
            prop.propagate_walkers_two_body(walkers, ham, trial)
        else:
            prop.propagate_walkers(walkers, ham, trial, trial.energy)
        if reortho:
            walkers.reortho()
        trial.calc_greens_function(walkers)

    return TestData(trial, walkers, ham, prop)


def build_driver_test_instance_fp(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    trial_type="phmsd",
    wfn_type="opt",
    complex_integrals: bool = False,
    complex_trial: bool = False,
    rhf_trial: bool = False,
    seed: Union[int, None] = None,
    density_diff=False,
    options: Union[dict, None] = None,
):
    if seed is not None:
        numpy.random.seed(seed)
    h1e, chol, _, _ = generate_hamiltonian(num_basis, num_elec, cplx=complex_integrals)
    system = Generic(nelec=num_elec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, num_basis**2)).T.copy(),
        ecore=0,
    )
    if density_diff:
        ham.density_diff = True
    trial, _ = build_random_trial(
        num_elec,
        num_basis,
        num_dets=num_dets,
        wfn_type=wfn_type,
        trial_type=trial_type,
        complex_trial=complex_trial,
        rhf_trial=rhf_trial,
    )
    trial.half_rotate(ham)
    try:
        trial.calculate_energy(system, ham)
    except NotImplementedError:
        pass

    qmc_opts = get_input_value(options, "qmc", default={}, alias=["qmc_options"])
    qmc = QMCOpts(qmc_opts, verbose=0)
    qmc.nwalkers = qmc.nwalkers
    afqmc = FPAFQMC.build(
        num_elec,
        ham,
        trial,
        num_walkers=qmc.nwalkers,
        seed=qmc.rng_seed,
        num_steps_per_block=5,
        num_blocks=2,
        timestep=qmc.dt,
        stabilize_freq=qmc.nstblz,
        pop_control_freq=qmc.npop_control,
        ene_0=trial.energy,
        num_iterations_fp=3,
    )
    return afqmc
