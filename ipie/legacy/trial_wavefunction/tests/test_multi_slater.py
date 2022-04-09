# disabling this test
# import numpy
# import scipy.linalg
# import pytest
# import sys
# from ipie.estimators.mixed import variational_energy_multi_det, local_energy
# from ipie.legacy.estimators.greens_function import gab
# from ipie.legacy.estimators.ci import get_hmatel, simple_fci
# from ipie.systems.generic import Generic
# from ipie.legacy.systems.ueg import UEG
# from ipie.utils.io import (
#         read_qmcpack_wfn_hdf,
#         write_qmcpack_wfn,
#         read_phfmol,
#         write_qmcpack_sparse
#         )
# from ipie.utils.misc import dotdict
# from ipie.utils.testing import get_random_wavefunction
# from ipie.trial_wavefunction.utils import get_trial_wavefunction
# from ipie.trial_wavefunction.multi_slater import MultiSlater
# from ipie.legacy.walkers.multi_det import MultiDetWalker

# @pytest.mark.unit
# def test_nomsd():
#     system = UEG({'nup': 7, 'ndown': 7, 'rs': 5, 'ecut': 4,
#                   'thermal': True})
#     import os
#     path = os.path.dirname(os.path.abspath(__file__))
#     wfn, psi0 = read_qmcpack_wfn_hdf(path+'/wfn.h5')
#     trial = MultiSlater(system, wfn, init=psi0)
#     trial.recompute_ci_coeffs(system)
#     # TODO: Fix python3.7 cython issue.
#     trial.calculate_energy(system)
#     ndets = trial.ndets
#     H = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
#     S = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
#     variational_energy_multi_det(system, trial.psi, trial.coeffs, H=H, S=S)
#     e, ev = scipy.linalg.eigh(H,S)
#     evar = variational_energy_multi_det(system, trial.psi, ev[:,0])
#     # TODO Check why this is wrong.
#     # assert e[0] == 0.15400990069739182
#     # assert e[0] == evar[0]

# # Todo: move to estimator tests.
# # def test_phmsd():
#     # coeff = numpy.array(eev[:,0], dtype=numpy.complex128)
#     # options = {'rediag': False}
#     # wfn = (numpy.array(coeff,dtype=numpy.complex128),numpy.array(oa),numpy.array(ob))
#     # numpy.random.seed(7)
#     # init = get_random_wavefunction(system.nelec, system.nbasis)
#     # na = system.nup
#     # trial = MultiSlater(system,  wfn, verbose=False,
#                         # options=options, init=init)
