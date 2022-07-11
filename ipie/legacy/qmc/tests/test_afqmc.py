import os

import numpy
import pytest
from mpi4py import MPI

from ipie.analysis.extraction import extract_mixed_estimates, extract_rdm
from ipie.legacy.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.qmc.afqmc import AFQMC
from ipie.legacy.systems.ueg import UEG
from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock
from ipie.qmc.calc import setup_calculation
from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian


@pytest.mark.driver
def test_constructor():
    options = {
        "verbosity": 0,
        "get_sha1": False,
        "qmc": {
            "timestep": 0.01,
            "num_steps": 10,
            "blocks": 10,
            "rng_seed": 8,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "walkers": {"pop_control": "comb"},
        "estimates": {
            "filename": "estimates.test_constructor.h5",
            "mixed": {"energy_eval_freq": 1},
        },
    }
    model = {
        "name": "UEG",
        "rs": 2.44,
        "ecut": 4,
        "nup": 7,
        "ndown": 7,
    }
    system = UEG(model)
    ham = HamUEG(system, model)
    trial = HartreeFock(system, ham, {})
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(
        comm=comm, options=options, system=system, hamiltonian=ham, trial=trial
    )
    afqmc.finalise(verbose=0)
    assert afqmc.trial.energy.real == pytest.approx(1.7796083856572522)


@pytest.mark.driver
def test_ueg():
    # FDM Updated benchmark to make run faster, old setup is commented out.
    options = {
        "verbosity": 0,
        "get_sha1": False,
        "qmc": {
            "timestep": 0.01,
            "num_steps": 10,
            "blocks": 5,
            "rng_seed": 8,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "system": {
            "name": "UEG",
            "rs": 2.44,
            "ecut": 2,
            "nup": 7,
            "ndown": 7,
        },
        "hamiltonian": {
            "name": "UEG",
        },
        "walkers": {"pop_control": "comb"},
        "estimates": {
            "filename": "estimates.test_ueg.h5",
            "mixed": {"energy_eval_freq": 1},
        },
        "trial": {"name": "hartree_fock"},
    }
    comm = MPI.COMM_WORLD
    # FDM: Fix cython issue.
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators["mixed"].update(
        afqmc.qmc, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.psi, 0
    )
    enum = afqmc.estimators.estimators["mixed"].names
    numer = afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
    # assert numer == pytest.approx(15.5272838037998)
    assert numer == pytest.approx(16.33039729324558)
    denom = afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
    assert denom == pytest.approx(10)
    weight = afqmc.estimators.estimators["mixed"].estimates[enum.uweight]
    # assert weight == pytest.approx(9.76035750882054)
    assert weight == pytest.approx(9.75405059997262)
    ehy = afqmc.psi.walkers[0].hybrid_energy
    # assert ehy.real == pytest.approx(2.41659997842609)
    assert ehy.real == pytest.approx(2.265850691148155)
    # assert ehy.imag == pytest.approx(-0.395817411848255)
    assert ehy.imag == pytest.approx(0.192627296991027)


@pytest.mark.driver
def test_hubbard():
    options = {
        "verbosity": 0,
        "get_sha1": False,
        "qmc": {
            "timestep": 0.01,
            "num_steps": 10,
            "blocks": 10,
            "rng_seed": 8,
            "nwalkers": 10,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "system": {
            "name": "Generic",
            "nup": 7,
            "ndown": 7,
        },
        "walkers": {"pop_control": "comb"},
        "hamiltonian": {
            "name": "Hubbard",
            "nx": 4,
            "ny": 4,
            "U": 4,
        },
        "trial": {"name": "hubbard_uhf"},
        "estimates": {
            "filename": "estimates.test_hubbard.h5",
            "mixed": {"energy_eval_freq": 1},
        },
        "propagator": {"hubbard_stratonovich": "discrete"},
    }
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators["mixed"].update(
        afqmc.qmc, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.psi, 0
    )
    enum = afqmc.estimators.estimators["mixed"].names
    numer = afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
    denom = afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
    weight = afqmc.estimators.estimators["mixed"].estimates[enum.weight]
    assert numer.real == pytest.approx(-152.68468568462666)
    data = extract_mixed_estimates("estimates.test_hubbard.h5")
    # old code seemed to ommit last value. Discard to avoid updating benchmark.
    assert numpy.mean(data.ETotal.values[:-1]) == pytest.approx(-14.974806533852874)


@pytest.mark.driver
def test_hubbard_complex():
    options = {
        "verbosity": 0,
        "get_sha1": False,
        "qmc": {
            "timestep": 0.01,
            "num_steps": 10,
            "blocks": 10,
            "rng_seed": 8,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "system": {
            "name": "Generic",
            "nup": 7,
            "ndown": 7,
        },
        "walkers": {"pop_control": "comb"},
        "hamiltonian": {
            "name": "Hubbard",
            "nx": 4,
            "ny": 4,
            "U": 4,
        },
        "trial": {"name": "hubbard_UHF"},
        "estimates": {
            "filename": "estimates.test_hubbard_complex.h5",
            "mixed": {"energy_eval_freq": 1},
        },
        "propagator": {"hubbard_stratonovich": "continuous"},
    }
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators["mixed"].update(
        afqmc.qmc, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.psi, 0
    )
    enum = afqmc.estimators.estimators["mixed"].names
    numer = afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
    denom = afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
    weight = afqmc.estimators.estimators["mixed"].estimates[enum.weight]
    assert numer == pytest.approx(-152.91937839611)
    data = extract_mixed_estimates("estimates.test_hubbard_complex.h5")
    assert numpy.mean(data.ETotal.values[:-1].real) == pytest.approx(-15.14323385684513)


@pytest.mark.driver
def test_generic():
    nmo = 11
    nelec = (3, 3)
    options = {
        "verbosity": 0,
        "get_sha1": False,
        "qmc": {
            "timestep": 0.005,
            "steps": 10,
            "blocks": 10,
            "rng_seed": 8,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "walkers": {"pop_control": "comb"},
        "estimates": {
            "filename": "estimates.test_generic.h5",
            "mixed": {"energy_eval_freq": 1},
        },
        "trial": {"name": "MultiSlater"},
    }
    numpy.random.seed(7)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, system=sys, hamiltonian=ham, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators["mixed"].update(
        afqmc.qmc, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.psi, 0
    )
    enum = afqmc.estimators.estimators["mixed"].names
    numer = afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
    denom = afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
    weight = afqmc.estimators.estimators["mixed"].estimates[enum.weight]
    assert numer.real == pytest.approx(3.8763193646854273)
    data = extract_mixed_estimates("estimates.test_generic.h5")
    assert numpy.mean(data.ETotal.values[:-1].real) == pytest.approx(1.5485077038208)


@pytest.mark.driver
def test_generic_single_det():
    nmo = 11
    nelec = (3, 3)
    options = {
        "verbosity": 0,
        "qmc": {
            "timestep": 0.005,
            "num_steps": 10,
            "blocks": 10,
            "rng_seed": 8,
            "batched": False,
            "pop_control_freq": 1,
            "stabilise_freq": 10,
        },
        "trial": {"name": "MultiSlater"},
        "walkers": {"pop_control": "comb"},
        "estimator": {
            "back_propagated": {"tau_bp": 0.025, "one_rdm": True},
            "filename": "estimates.test_generic_single_det.h5",
            "mixed": {"energy_eval_freq": 1},
        },
    }
    numpy.random.seed(7)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys_opts = {"sparse": True}
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )
    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, system=sys, hamiltonian=ham, options=options)
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators["mixed"].update(
        afqmc.qmc, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.psi, 0
    )
    enum = afqmc.estimators.estimators["mixed"].names
    numer = afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
    denom = afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
    weight = afqmc.estimators.estimators["mixed"].estimates[enum.weight]
    assert numer.real == pytest.approx(3.8763193646854273)
    data = extract_mixed_estimates("estimates.test_generic_single_det.h5")
    assert numpy.mean(data.ETotal.values[:-1].real) == pytest.approx(1.5485077038208)
    rdm = extract_rdm("estimates.test_generic_single_det.h5")
    assert rdm[0, 0].trace() == pytest.approx(nelec[0])
    assert rdm[0, 1].trace() == pytest.approx(nelec[1])
    assert rdm[11, 0, 1, 3].real == pytest.approx(-0.121883381144845)


def teardown_module(self):
    cwd = os.getcwd()
    files = [
        "estimates.test_generic_single_det.h5",
        "estimates.test_generic.h5",
        "estimates.test_constructor.h5",
        "estimates.test_ueg.h5",
        "estimates.test_hubbard.h5",
        "estimates.test_hubbard_complex.h5",
        "estimates.test_hubbard_complex.h5",
    ]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass
