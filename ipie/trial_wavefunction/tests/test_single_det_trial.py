import numpy as np
import pytest
from mpi4py import MPI

from ipie.utils.testing import (
        get_random_nomsd,
        get_random_nomsd_ghf,
        get_random_sys_ham
        )
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF


@pytest.mark.unit
def test_single_det():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    np.random.seed(7)
    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    trial = SingleDet(
        wavefunction[1][0],
        (nalpha, nbeta),
        nbasis,
    )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(wavefunction[0])
    trial.build()
    comm = MPI.COMM_WORLD
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial.half_rotate(ham, comm=comm)
    assert trial._rchola.shape == (naux, nbasis * nalpha)
    assert trial._rcholb.shape == (naux, nbasis * nbeta)
    assert trial._rH1a.shape == (nalpha, nbasis)
    assert trial._rH1b.shape == (nbeta, nbasis)

@pytest.mark.unit
def test_single_det_ghf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    np.random.seed(7)

    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    trial = SingleDet(
            wavefunction[1][0],
            (nalpha, nbeta),
            nbasis,
            )
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial.half_rotate(ham)
    trial.calculate_energy(sys, ham)

    print("trial.energy = {}".format(trial.energy))


    wavefunction = get_random_nomsd_ghf(nalpha, nbeta, nbasis, ndet=1)
    trial = SingleDetGHF(
            wavefunction[1][0],
            (nalpha, nbeta),
            nbasis*2,
            )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis*2
    assert trial.num_dets == len(wavefunction[0])

#     trial.build()
#     comm = MPI.COMM_WORLD
#     sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
#     trial.half_rotate(ham, comm=comm)
#     assert trial._rchola.shape == (naux, nbasis*nalpha)
#     assert trial._rcholb.shape == (naux, nbasis*nbeta)
#     assert trial._rH1a.shape == (nalpha, nbasis)
#     assert trial._rH1b.shape == (nbeta, nbasis)

# def __main__():
test_single_det()
test_single_det_ghf()