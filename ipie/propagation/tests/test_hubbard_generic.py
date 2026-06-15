from types import SimpleNamespace

import numpy
import pytest

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers


def _random_orthonormal(nbasis, nocc, rng):
    mat = rng.normal(size=(nbasis, nocc)) + 1.0j * rng.normal(size=(nbasis, nocc))
    q, _ = numpy.linalg.qr(mat)
    return numpy.ascontiguousarray(q[:, :nocc])


def _copy_walkers(walkers):
    return SimpleNamespace(
        nwalkers=walkers.nwalkers,
        nup=walkers.nup,
        ndown=walkers.ndown,
        nbasis=walkers.nbasis,
        rhf=walkers.rhf,
        weight=walkers.weight.copy(),
        ovlp=walkers.ovlp.copy(),
        phia=walkers.phia.copy(),
        phib=walkers.phib.copy(),
        inv_ovlp_a=walkers.inv_ovlp_a.copy(),
        inv_ovlp_b=walkers.inv_ovlp_b.copy(),
    )


def _build_case(rhf):
    rng = numpy.random.default_rng(7)
    nbasis = 12
    nup = 5
    ndown = 5 if rhf else 4
    nwalkers = 3

    h1 = rng.normal(size=(2, nbasis, nbasis))
    h1 = 0.5 * (h1 + h1.transpose(0, 2, 1))
    hamiltonian = Hubbard(h1, U=4.0)

    psi0a = _random_orthonormal(nbasis, nup, rng)
    psi0b = _random_orthonormal(nbasis, ndown, rng)
    psi = numpy.ascontiguousarray(numpy.hstack([psi0a, psi0b]))

    handler = MPIHandler()
    trial = SingleDet(psi, (nup, ndown), nbasis, handler=handler, verbose=False)
    trial.half_rotate(hamiltonian, handler.scomm)
    walkers = UHFWalkers(psi, nup, ndown, nbasis, nwalkers, handler)
    walkers.build(trial)
    walkers.rhf = rhf

    prop = HubbardSingleSite(0.005)
    prop.build(hamiltonian, trial, walkers, handler)
    random_fields = rng.random((nbasis, nwalkers))
    return prop, hamiltonian, trial, walkers, random_fields


@pytest.mark.unit
@pytest.mark.parametrize("rhf", [False, True])
def test_hubbard_single_site_cpu_numba_matches_legacy(rhf):
    prop, hamiltonian, trial, walkers, random_fields = _build_case(rhf)
    ref = _copy_walkers(walkers)
    opt = _copy_walkers(walkers)

    prop._legacy_two_body(ref, hamiltonian, trial, random_fields=random_fields)
    prop._cpu_numba_two_body(opt, hamiltonian, trial, random_fields)

    numpy.testing.assert_allclose(opt.weight, ref.weight, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.ovlp, ref.ovlp, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.phia, ref.phia, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.phib, ref.phib, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.inv_ovlp_a, ref.inv_ovlp_a, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.inv_ovlp_b, ref.inv_ovlp_b, rtol=1.0e-10, atol=1.0e-10)
