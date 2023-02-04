import numpy as np
import pytest
from mpi4py import MPI

from ipie.utils.testing import get_random_nomsd, get_random_sys_ham
from ipie.trial_wavefunction.noci import NOCI


@pytest.mark.unit
def test_noci():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    np.random.seed(7)
    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=15)
    trial = NOCI(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
    )
    assert trial.nelec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(wavefunction[0])
    trial.build()
    trial.num_dets = 10
    trial.build()
    assert trial.num_dets == 10
    comm = MPI.COMM_WORLD
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial.half_rotate(sys, ham, comm=comm)
    assert trial._rchola.shape == (trial.num_dets, naux, nbasis * nalpha)
    assert trial._rcholb.shape == (trial.num_dets, naux, nbasis * nbeta)
    assert trial._rH1a.shape == (trial.num_dets, nalpha, nbasis)
    assert trial._rH1b.shape == (trial.num_dets, nbeta, nbasis)
