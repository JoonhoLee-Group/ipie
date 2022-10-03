import sys
import numpy as np

use_gpu = bool(int(sys.argv[1]))
from ipie.config import config
config.update_option('use_gpu', use_gpu)

na = 15
nb = 15
nmo = 30
naux = 10*nmo

from ipie.utils.testing import get_random_phmsd_opt, shaped_normal

ndets = [10, 100, 200, 250]

from ipie.estimators.kernels import wicks as wk
fnc_map = [
    wk.get_dets_singles,
    wk.get_dets_doubles,
    wk.get_dets_triples,
    wk.get_dets_nfold,
    ]

for nd in ndets:
    dist = np.zeros((nmo), dtype=np.int32)
    dist[:5] = nd // 4
    wfn, init = get_random_phmsd_opt(na, nb, nmo, dist=[dist, dist], init=True)
    from ipie.systems import Generic
    system = Generic(nelec=(na, na))
    chol = shaped_normal((naux, nmo, nmo))

    from ipie.hamiltonians import Generic as HamGeneric
    h1e = shaped_normal((nmo, nmo))
    ham = HamGeneric(
        h1e=np.array([h1e, h1e]),
        chol=chol.reshape((naux, nmo * nmo)).T.copy(),
        h1e_mod=h1e.copy(),
        ecore=0,
        verbose=False,
    )

    from ipie.trial_wavefunction import MultiSlater
    trial = MultiSlater(
            system,
            ham,
            wfn,
            options={"compute_trial_energy": False},
            init=init,
            verbose=False
            )

    nwalkers = 100

    G0 = shaped_normal((nwalkers, nmo, nmo), cmplx=True)
    from ipie.utils.backend import arraylib as xp
    for iex, ex in enumerate([1, 2, 3, 4]):
        dets = xp.zeros((nwalkers, trial.cre_ex_a[ex].shape[0]), dtype=xp.complex128)
        if ex == 4:
            det_mat_buffer = xp.zeros((nwalkers, trial.cre_ex_a[ex].shape[0], 4, 4), dtype=xp.complex128)
        ndets = dets.shape[1]
        ex_a = xp.array(trial.cre_ex_a[ex])
        anh_a = xp.array(trial.anh_ex_a[ex])
        occ_a = xp.array(trial.occ_map_a)
        if ex == 4 and use_gpu:
            fnc_map[iex](ex_a, anh_a, occ_a, 0, G0, det_mat_buffer, dets)
        else:
            fnc_map[iex](ex_a, anh_a, occ_a, 0, G0, dets)
        import time
        av_time = 0
        for i in range(5):
            start = time.time()
            if ex == 4 and use_gpu:
                fnc_map[iex](ex_a, anh_a, occ_a, 0, G0, det_mat_buffer, dets)
            else:
                fnc_map[iex](ex_a, anh_a, occ_a, 0, G0, dets)
            av_time += time.time()-start
        print(ndets, ex, av_time/5)
