import numpy
from mpi4py import MPI
import os
import pytest
from ipie.analysis.extraction import (
        extract_mixed_estimates,
        extract_rdm
        )
from ipie.legacy.qmc.afqmc import AFQMC
from ipie.legacy.systems.ueg import UEG
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock

from ipie.qmc.calc import setup_calculation
from ipie.qmc.afqmc_batch import AFQMCBatch
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians.generic import Generic as LegacyHamGeneric
from ipie.utils.testing import generate_hamiltonian

steps = 25
blocks = 7
seed = 7
nwalkers = 15
nmo = 14
nelec = (4,3)
pop_control_freq = 5
stabilise_freq = 5

@pytest.mark.driver
def test_generic_single_det_batch():
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.005,
                'steps': steps,
                'nwalkers_per_task':nwalkers,
                'pop_control_freq': pop_control_freq,
                'stabilise_freq': stabilise_freq,
                'blocks': blocks,
                'rng_seed': seed,
                'batched': True
            },
            'estimates': {
                'mixed': {
                    'energy_eval_freq': 1
                }
            },
            'trial': {
                'name': 'MultiSlater'
            },
            'walkers': {
                'population_control':'pair_branch'
            }
        }
    numpy.random.seed(seed)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec) 
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    comm = MPI.COMM_WORLD
    afqmc = AFQMCBatch(comm=comm, system=sys, hamiltonian = ham, options=options)
    afqmc.estimators.estimators['mixed'].print_header()
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators['mixed'].update_batch(afqmc.qmc, afqmc.system, afqmc.hamiltonian,
                                                afqmc.trial, afqmc.psi.walkers_batch, 0)
    enum_batch = afqmc.estimators.estimators['mixed'].names
    numer_batch = afqmc.estimators.estimators['mixed'].estimates[enum_batch.enumer]
    denom_batch = afqmc.estimators.estimators['mixed'].estimates[enum_batch.edenom]
    weight_batch = afqmc.estimators.estimators['mixed'].estimates[enum_batch.weight]

    data_batch = extract_mixed_estimates('estimates.0.h5')

    numpy.random.seed(seed)
    options = {
            'verbosity': 0,
            'get_sha1': False,
            'qmc': {
                'timestep': 0.005,
                'steps': steps,
                'nwalkers_per_task':nwalkers,
                'pop_control_freq':pop_control_freq,
                'stabilise_freq': stabilise_freq,
                'blocks': blocks,
                'rng_seed': seed,
                'batched': False
            },
            'estimates': {
                'mixed': {
                    'energy_eval_freq': 1
                }
            },
            'trial': {
                'name': 'MultiSlater'
            },
            'walkers': {
                'population_control':'pair_branch'
            }
        }
    numpy.random.seed(seed)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec) 
    legacyham = LegacyHamGeneric(h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)

    comm = MPI.COMM_WORLD
    afqmc = AFQMC(comm=comm, system=sys, hamiltonian = legacyham, options=options)
    afqmc.estimators.estimators['mixed'].print_header()
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.estimators['mixed'].update(afqmc.qmc, afqmc.system, afqmc.hamiltonian,
                                                afqmc.trial, afqmc.psi, 0)
    enum = afqmc.estimators.estimators['mixed'].names
    numer = afqmc.estimators.estimators['mixed'].estimates[enum.enumer]
    denom = afqmc.estimators.estimators['mixed'].estimates[enum.edenom]
    weight = afqmc.estimators.estimators['mixed'].estimates[enum.weight]

    assert numer.real == pytest.approx(numer_batch.real)
    assert denom.real == pytest.approx(denom_batch.real)
    assert weight.real == pytest.approx(weight_batch.real)
    assert numer.imag == pytest.approx(numer_batch.imag)
    assert denom.imag == pytest.approx(denom_batch.imag)
    assert weight.imag == pytest.approx(weight_batch.imag)
    data = extract_mixed_estimates('estimates.0.h5')

    assert numpy.mean(data_batch.WeightFactor.values[:-1].real) == pytest.approx(numpy.mean(data.WeightFactor.values[:-1].real))
    assert numpy.mean(data_batch.Weight.values[:-1].real) == pytest.approx(numpy.mean(data.Weight.values[:-1].real))
    assert numpy.mean(data_batch.ENumer.values[:-1].real) == pytest.approx(numpy.mean(data.ENumer.values[:-1].real))
    assert numpy.mean(data_batch.EDenom.values[:-1].real) == pytest.approx(numpy.mean(data.EDenom.values[:-1].real))
    assert numpy.mean(data_batch.ETotal.values[:-1].real) == pytest.approx(numpy.mean(data.ETotal.values[:-1].real))
    assert numpy.mean(data_batch.E1Body.values[:-1].real) == pytest.approx(numpy.mean(data.E1Body.values[:-1].real))
    assert numpy.mean(data_batch.E2Body.values[:-1].real) == pytest.approx(numpy.mean(data.E2Body.values[:-1].real))
    assert numpy.mean(data_batch.EHybrid.values[:-1].real) == pytest.approx(numpy.mean(data.EHybrid.values[:-1].real))
    assert numpy.mean(data_batch.Overlap.values[:-1].real) == pytest.approx(numpy.mean(data.Overlap.values[:-1].real))

if __name__=="__main__":
    test_generic_single_det_batch()
