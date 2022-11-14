import numpy as np
import pytest

from ipie.utils.backend import arraylib as xp
from ipie.estimators.kernels import wicks as wk
from ipie.utils.testing import (
        gen_random_test_instances,
        shaped_normal)

np.random.seed(7)
nmo = 10
nelec = (3, 3)
naux = 100
nwalker = 10
ndets = 1000
sys, ham, walkers, trial = gen_random_test_instances(
    nmo, nelec, naux, nwalker, wfn_type="phmsd", ndets=ndets
)
# trial.cast_to_cupy()


@pytest.mark.gpu
def test_get_dets_single_excitation_batched():
    CI = np.zeros((nwalker, nmo, nmo))
    ndet_single_a = len(trial.cre_ex_a[1])
    phases = shaped_normal((ndet_single_a,), cmplx=True)
    pqs = [str(p[0])+str(q[0]) for p, q in zip(trial.cre_ex_a[1], trial.anh_ex_a[1])]
    print(len(pqs), len(set(pqs)))
    # for iw in range(nwalker):
        # for idet in range(ndet_single_a):
            # p = trial.cre_ex_a[1][idet]
            # q = trial.anh_ex_a[1][idet]


if __name__ == "__main__":
    test_get_dets_single_excitation_batched()
