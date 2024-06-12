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

import pytest
import numpy

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_ueg_test_case_handlers
    from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers
    from ipie.legacy.estimators.ueg import local_energy_ueg as legacy_local_energy_ueg

    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.utils.testing import build_ueg_test_case_handlers

from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G

comm = MPI.COMM_WORLD


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_phaseless_ueg_propagator():
    # UEG params.
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    rs = 1.0
    ecut = 1.0

    # Thermal AFQMC params.
    mu = -1.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    lowrank = False

    debug = True
    verbose = False if (comm.rank != 0) else True
    seed = 7
    numpy.random.seed(seed)

    # Test.
    objs = build_ueg_test_case_handlers(
        nelec,
        rs,
        ecut,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        debug=debug,
        seed=seed,
        verbose=verbose,
    )
    trial = objs["trial"]
    hamiltonian = objs["hamiltonian"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]

    # Legacy.
    legacy_objs = build_legacy_ueg_test_case_handlers(
        comm,
        nelec,
        rs,
        ecut,
        mu,
        beta,
        timestep,
        nwalkers=nwalkers,
        lowrank=lowrank,
        seed=seed,
        verbose=verbose,
    )
    legacy_system = legacy_objs["system"]
    legacy_trial = legacy_objs["trial"]
    legacy_hamiltonian = legacy_objs["hamiltonian"]
    legacy_walkers = legacy_objs["walkers"]
    legacy_propagator = legacy_objs["propagator"]

    h1e = legacy_hamiltonian.H1[0]
    eri = legacy_hamiltonian.eri_4()

    for t in range(walkers.stack[0].nslice):
        for iw in range(walkers.nwalkers):
            P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]]))
            eloc = local_energy_generic_cholesky(hamiltonian, P)

            legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
            legacy_eloc = legacy_local_energy_ueg(legacy_system, legacy_hamiltonian, legacy_P)

            legacy_Pa, legacy_Pb = legacy_P
            legacy_Ptot = legacy_Pa + legacy_Pb
            ref_e1 = numpy.einsum("ij,ij->", h1e, legacy_Ptot)

            Ptot = legacy_Ptot
            Pa = legacy_Pa
            Pb = legacy_Pb

            ecoul = 0.5 * numpy.einsum("ijkl,ij,kl->", eri, Ptot, Ptot)
            exx = -0.5 * numpy.einsum("ijkl,il,kj->", eri, Pa, Pa)
            exx -= 0.5 * numpy.einsum("ijkl,il,kj->", eri, Pb, Pb)
            ref_e2 = ecoul + exx
            ref_eloc = (ref_e1 + ref_e2, ref_e1, ref_e2)

            numpy.testing.assert_almost_equal(legacy_P, P, decimal=10)
            numpy.testing.assert_almost_equal(legacy_trial.dmat, trial.dmat, decimal=10)
            numpy.testing.assert_allclose(eloc, ref_eloc, atol=1e-10)
            numpy.testing.assert_allclose(legacy_eloc, ref_eloc, atol=1e-10)
            numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)

            numpy.testing.assert_almost_equal(
                legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10
            )
            numpy.testing.assert_almost_equal(
                legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10
            )
            numpy.testing.assert_almost_equal(
                legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10
            )
            numpy.testing.assert_almost_equal(
                legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10
            )

        propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
        legacy_walkers = legacy_propagate_walkers(
            legacy_hamiltonian, legacy_trial, legacy_walkers, legacy_propagator, xi=propagator.xi
        )


if __name__ == "__main__":
    test_phaseless_ueg_propagator()
