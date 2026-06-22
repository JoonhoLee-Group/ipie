"""Tests for the multi-determinant (NOCI) GHF trial + discrete-Hirsch propagation.

A one-determinant NOCIGHF must reproduce SingleDetGHF (overlap, Green's function,
local energy, and a full two-body step) bit-for-bit, and the multi-determinant
trial energy must agree with an independent NOCI evaluation.
"""
import numpy
import pytest

from ipie.hamiltonians.hubbard import Hubbard
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.trial_wavefunction.noci_ghf import NOCIGHF
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.estimators.energy import local_energy


def _hubbard_T(nx, ny, t=1.0):
    """Complex (Peierls-phase) nearest-neighbour hopping on an nx*ny lattice.

    Complex hopping makes T Hermitian but non-symmetric, so the one-body energy
    contraction must be "ij,ij" on the ipie-convention Green's function (and
    "ij,ji" on its transpose) -- a real-T test would not catch a transpose bug.
    """
    n = nx * ny
    T = numpy.zeros((n, n), dtype=numpy.complex128)
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            if x + 1 < nx:
                j = (x + 1) * ny + y
                T[i, j] = -t * numpy.exp(0.3j)
                T[j, i] = numpy.conj(T[i, j])
            if y + 1 < ny:
                j = x * ny + (y + 1)
                T[i, j] = -t * numpy.exp(0.3j)
                T[j, i] = numpy.conj(T[i, j])
    return T


def _rand_det(rng, nb, nocc):
    M = rng.standard_normal((2 * nb, nocc)) + 1j * rng.standard_normal((2 * nb, nocc))
    Q, _ = numpy.linalg.qr(M)
    return Q[:, :nocc]


@pytest.fixture
def setup():
    rng = numpy.random.default_rng(7)
    nx, ny, nup, ndown = 2, 2, 1, 1
    nb, nocc = nx * ny, nup + ndown
    T = _hubbard_T(nx, ny)
    ham = Hubbard(numpy.array([T, T]), U=4.0)
    syst = Generic(nelec=(nup, ndown))
    return rng, ham, syst, nb, nocc, nup, ndown


@pytest.mark.unit
def test_noci_ghf_k1_parity(setup):
    rng, ham, syst, nb, nocc, nup, ndown = setup
    psi0 = _rand_det(rng, nb, nocc)
    nwalkers = 5
    trial_sd = SingleDetGHF(psi0.copy(), (nup, ndown), nb)
    trial_n1 = NOCIGHF(psi0.copy()[None], numpy.array([1.0 + 0j]), (nup, ndown), nb)

    init = _rand_det(rng, nb, nocc)
    w_sd = GHFWalkers(init.copy(), nup, ndown, nb, nwalkers)
    w_n1 = GHFWalkers(init.copy(), nup, ndown, nb, nwalkers)
    w_sd.build(trial_sd)
    w_n1.build(trial_n1)

    assert numpy.allclose(w_sd.ovlp, w_n1.ovlp, atol=1e-12)
    assert numpy.allclose(w_sd.G, w_n1.G, atol=1e-12)
    e_sd = local_energy(syst, ham, w_sd, trial_sd)
    e_n1 = local_energy(syst, ham, w_n1, trial_n1)
    assert numpy.allclose(e_sd, e_n1, atol=1e-10)

    prop = HubbardSingleSite(0.01)
    prop.build(ham)
    rfields = rng.random((nb, nwalkers))
    w_sd.inverse_overlap(trial_sd)
    prop._ghf_two_body(w_sd, ham, trial_sd, rfields.copy())
    prop._noci_ghf_two_body(w_n1, ham, trial_n1, rfields.copy())
    assert numpy.allclose(w_sd.phi, w_n1.phi, atol=1e-11)
    assert numpy.allclose(w_sd.weight, w_n1.weight, atol=1e-11)
    assert numpy.allclose(w_sd.ovlp, w_n1.ovlp, atol=1e-11)


@pytest.mark.unit
def test_noci_ghf_multidet_energy(setup):
    rng, ham, syst, nb, nocc, nup, ndown = setup
    T = ham.T[0]
    U = ham.U
    K = 4
    dets = numpy.array([_rand_det(rng, nb, nocc) for _ in range(K)])
    coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    trial = NOCIGHF(dets, coeffs, (nup, ndown), nb)
    trial.calculate_energy(syst, ham)

    # independent full double-sum NOCI variational energy <Psi|H|Psi>/<Psi|Psi>
    num = den = 0.0 + 0j
    for k in range(K):
        for l in range(K):
            A = dets[k].conj().T @ dets[l]
            O = numpy.linalg.det(A)
            G = dets[l] @ numpy.linalg.solve(A, dets[k].conj().T)
            Gaa, Gbb = G[:nb, :nb], G[nb:, nb:]
            Gab, Gba = G[:nb, nb:], G[nb:, :nb]
            e1 = numpy.einsum("ij,ji", T, Gaa) + numpy.einsum("ij,ji", T, Gbb)
            eU = U * numpy.sum(numpy.diag(Gaa) * numpy.diag(Gbb)
                               - numpy.diag(Gab) * numpy.diag(Gba))
            w = numpy.conj(coeffs[k]) * coeffs[l] * O
            num += w * (e1 + eU)
            den += w
    assert abs(complex(trial.energy) - num / den) < 1e-10
