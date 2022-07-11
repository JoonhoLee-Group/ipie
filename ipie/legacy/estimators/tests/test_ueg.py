import numpy
import pytest

from ipie.legacy.estimators.greens_function import gab
from ipie.legacy.estimators.ueg import fock_ueg, local_energy_ueg
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.systems.ueg import UEG
from ipie.utils.misc import timeit
from ipie.utils.testing import get_random_wavefunction


@pytest.mark.unit
def test_fock_build():
    sys = UEG({"rs": 2.0, "ecut": 2, "nup": 7, "ndown": 7, "thermal": True})
    ham = HamUEG(sys, {"rs": 2.0, "ecut": 2.5, "nup": 7, "ndown": 7})
    numpy.random.seed(7)
    psi = get_random_wavefunction(sys.nelec, ham.nbasis).real
    trial = numpy.eye(ham.nbasis, sys.nelec[0])
    G = numpy.array(
        [
            gab(psi[:, : sys.nup], psi[:, : sys.nup]),
            gab(psi[:, sys.nup :], psi[:, sys.nup :]),
        ]
    ).astype(numpy.complex128)
    nb = ham.nbasis
    eris = ham.eri_4()
    F = fock_ueg(ham, G)
    vj = numpy.einsum("pqrs,xqp->xrs", eris, G)
    vk = numpy.einsum("pqrs,xqr->xps", eris, G)
    fock = numpy.zeros((2, 33, 33), dtype=numpy.complex128)
    fock[0] = ham.H1[0] + vj[0] + vj[1] - vk[0]
    fock[1] = ham.H1[1] + vj[0] + vj[1] - vk[1]
    assert numpy.linalg.norm(fock - F) == pytest.approx(0.0)


@pytest.mark.unit
def test_build_J():
    # sys = UEG({'rs': 2.0, 'ecut': 2.0, 'nup': 7, 'ndown': 7, 'thermal': True})
    sys = UEG({"rs": 2.0, "ecut": 2, "nup": 7, "ndown": 7, "thermal": True})
    ham = HamUEG(sys, {"rs": 2.0, "ecut": 2.5, "nup": 7, "ndown": 7})
    Gkpq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    psi = get_random_wavefunction(sys.nelec, ham.nbasis).real
    trial = numpy.eye(ham.nbasis, sys.nelec[0])
    G = numpy.array(
        [
            gab(psi[:, : sys.nup], psi[:, : sys.nup]),
            gab(psi[:, sys.nup :], psi[:, sys.nup :]),
        ]
    )
    from ipie.legacy.estimators.ueg import coulomb_greens_function

    for s in [0, 1]:
        coulomb_greens_function(
            len(ham.qvecs),
            ham.ikpq_i,
            ham.ikpq_kpq,
            ham.ipmq_i,
            ham.ipmq_pmq,
            Gkpq[s],
            Gpmq[s],
            G[s],
        )

    from ipie.legacy.estimators.ueg import build_J

    J1 = timeit(build_J)(ham, Gpmq, Gkpq)
    from ipie.legacy.estimators.ueg_kernels import build_J_opt

    J2 = timeit(build_J_opt)(
        len(ham.qvecs),
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        Gkpq,
        Gpmq,
    )
    assert numpy.linalg.norm(J1 - J2) == pytest.approx(0.0)


@pytest.mark.unit
def test_build_K():
    sys = UEG({"rs": 2.0, "ecut": 2.0, "nup": 7, "ndown": 7, "thermal": True})
    ham = HamUEG(sys, {"rs": 2.0, "ecut": 2.5, "nup": 7, "ndown": 7})
    Gkpq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    psi = get_random_wavefunction(sys.nelec, ham.nbasis).real
    trial = numpy.eye(ham.nbasis, sys.nelec[0])
    G = numpy.array(
        [
            gab(psi[:, : sys.nup], psi[:, : sys.nup]),
            gab(psi[:, sys.nup :], psi[:, sys.nup :]),
        ]
    ).astype(numpy.complex128)
    from ipie.legacy.estimators.ueg import build_K
    from ipie.legacy.estimators.ueg_kernels import build_K_opt

    K1 = timeit(build_K)(ham, G)
    K2 = timeit(build_K_opt)(
        len(ham.qvecs),
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        G,
    )
    assert numpy.linalg.norm(K1 - K2) == pytest.approx(0.0)
