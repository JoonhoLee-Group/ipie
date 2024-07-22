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
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

import numpy
import pytest
from typing import Union

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers_mpi
    from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers

    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.addons.thermal.walkers.pop_controller import ThermalPopController
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers_mpi

comm = MPI.COMM_WORLD


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_pair_branch_batch():
    mpi_handler = MPIHandler()

    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 12
    nblocks = 3
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    pop_control_method = "pair_branch"
    lowrank = False

    mf_trial = True
    complex_integrals = False
    debug = True
    verbose = False if (comm.rank != 0) else True
    seed = 7
    numpy.random.seed(seed)

    # Test.
    objs = build_generic_test_case_handlers_mpi(
        nelec,
        nbasis,
        mu,
        beta,
        timestep,
        mpi_handler,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        complex_integrals=complex_integrals,
        debug=debug,
        seed=seed,
        verbose=verbose,
    )
    trial = objs["trial"]
    hamiltonian = objs["hamiltonian"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]
    pcontrol = ThermalPopController(
        nwalkers, nsteps_per_block, mpi_handler, pop_control_method, verbose=verbose
    )

    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
        hamiltonian,
        mpi_handler,
        nelec,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        seed=seed,
        verbose=verbose,
    )
    legacy_system = legacy_objs["system"]
    legacy_trial = legacy_objs["trial"]
    legacy_hamiltonian = legacy_objs["hamiltonian"]
    legacy_walkers = legacy_objs["walkers"]
    legacy_propagator = legacy_objs["propagator"]

    for block in range(nblocks):
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                legacy_hamiltonian,
                legacy_trial,
                legacy_walkers,
                legacy_propagator,
                xi=propagator.xi,
            )

            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial)  # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(
            walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight
        )


# TODO: Lowrank code is WIP. See: https://github.com/JoonhoLee-Group/ipie/issues/302
# @pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
# @pytest.mark.unit
def test_pair_branch_batch_lowrank():
    mpi_handler = MPIHandler()

    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 12
    nblocks = 3
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    pop_control_method = "pair_branch"
    lowrank = True

    mf_trial = False
    diagonal = True
    complex_integrals = False
    debug = True
    verbose = False if (comm.rank != 0) else True
    seed = 7
    numpy.random.seed(seed)

    options = {
        "nelec": nelec,
        "nbasis": nbasis,
        "mu": mu,
        "beta": beta,
        "timestep": timestep,
        "nwalkers": nwalkers,
        "seed": seed,
        "nsteps_per_block": nsteps_per_block,
        "nblocks": nblocks,
        "stabilize_freq": stabilize_freq,
        "pop_control_freq": pop_control_freq,
        "pop_control_method": pop_control_method,
        "lowrank": lowrank,
        "complex_integrals": complex_integrals,
        "mf_trial": mf_trial,
        "propagate": propagate,
        "diagonal": diagonal,
    }

    # Test.
    objs = build_generic_test_case_handlers_mpi(
        nelec,
        nbasis,
        mu,
        beta,
        timestep,
        mpi_handler,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        complex_integrals=complex_integrals,
        diagonal=diagonal,
        debug=debug,
        seed=seed,
        verbose=verbose,
    )
    trial = objs["trial"]
    hamiltonian = objs["hamiltonian"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]
    pcontrol = ThermalPopController(
        nwalkers,
        nsteps_per_block,
        mpi_handler,
        pop_control_method=pop_control_method,
        verbose=verbose,
    )

    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
        hamiltonian,
        mpi_handler,
        nelec,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        seed=seed,
        verbose=verbose,
    )
    legacy_system = legacy_objs["system"]
    legacy_trial = legacy_objs["trial"]
    legacy_hamiltonian = legacy_objs["hamiltonian"]
    legacy_walkers = legacy_objs["walkers"]
    legacy_propagator = legacy_objs["propagator"]

    for block in range(nblocks):
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                legacy_hamiltonian,
                legacy_trial,
                legacy_walkers,
                legacy_propagator,
                xi=propagator.xi,
            )

            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial)  # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(
            walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight
        )


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_comb_batch():
    mpi_handler = MPIHandler()
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 12
    nblocks = 3
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    pop_control_method = "comb"
    lowrank = False

    mf_trial = True
    complex_integrals = False
    debug = True
    verbose = False if (comm.rank != 0) else True
    seed = 7
    numpy.random.seed(seed)

    # Test.
    objs = build_generic_test_case_handlers_mpi(
        nelec,
        nbasis,
        mu,
        beta,
        timestep,
        mpi_handler,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        complex_integrals=complex_integrals,
        debug=debug,
        seed=seed,
        verbose=verbose,
    )
    trial = objs["trial"]
    hamiltonian = objs["hamiltonian"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]
    pcontrol = ThermalPopController(
        nwalkers,
        nsteps_per_block,
        mpi_handler,
        pop_control_method=pop_control_method,
        verbose=verbose,
    )

    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
        hamiltonian,
        mpi_handler,
        nelec,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        pop_control_method=pop_control_method,
        seed=seed,
        verbose=verbose,
    )
    legacy_system = legacy_objs["system"]
    legacy_trial = legacy_objs["trial"]
    legacy_hamiltonian = legacy_objs["hamiltonian"]
    legacy_walkers = legacy_objs["walkers"]
    legacy_propagator = legacy_objs["propagator"]

    for block in range(nblocks):
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                legacy_hamiltonian,
                legacy_trial,
                legacy_walkers,
                legacy_propagator,
                xi=propagator.xi,
            )

            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial)  # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(
            walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight
        )


# TODO: Lowrank code is WIP. See: https://github.com/JoonhoLee-Group/ipie/issues/302
# @pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
# @pytest.mark.unit
def test_comb_batch_lowrank():
    mpi_handler = MPIHandler()

    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 12
    nblocks = 3
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    pop_control_method = "comb"
    lowrank = True

    mf_trial = False
    diagonal = True
    complex_integrals = False
    debug = True
    verbose = False if (comm.rank != 0) else True
    seed = 7
    numpy.random.seed(seed)

    # Test.
    objs = build_generic_test_case_handlers_mpi(
        nelec,
        nbasis,
        mu,
        beta,
        timestep,
        mpi_handler,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        complex_integrals=complex_integrals,
        diagonal=diagonal,
        debug=debug,
        seed=seed,
        verbose=verbose,
    )
    trial = objs["trial"]
    hamiltonian = objs["hamiltonian"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]
    pcontrol = ThermalPopController(
        nwalkers,
        nsteps_per_block,
        mpi_handler,
        pop_control_method=pop_control_method,
        verbose=verbose,
    )

    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
        hamiltonian,
        mpi_handler,
        nelec,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        mf_trial=mf_trial,
        seed=seed,
        verbose=verbose,
    )
    legacy_system = legacy_objs["system"]
    legacy_trial = legacy_objs["trial"]
    legacy_hamiltonian = legacy_objs["hamiltonian"]
    legacy_walkers = legacy_objs["walkers"]
    legacy_propagator = legacy_objs["propagator"]

    for block in range(nblocks):
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                legacy_hamiltonian,
                legacy_trial,
                legacy_walkers,
                legacy_propagator,
                xi=propagator.xi,
            )

            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial)  # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(
            walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight
        )


if __name__ == "__main__":
    test_pair_branch_batch()
    test_comb_batch()

    # test_pair_branch_batch_lowrank()
    # test_comb_batch_lowrank()
