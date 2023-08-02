import numpy
import pytest
import scipy

from ipie.propagation.operations import (
    apply_exponential,
    apply_exponential_batch,
    propagate_one_body,
)


@pytest.mark.unit
def test_propagate_one_body():
    nwalkers = 10
    nbasis = 25
    nocc = 10

    ndim = nwalkers * nbasis * nocc
    phi = numpy.random.randn(ndim) + 1.0j * numpy.random.randn(ndim)

    phi = phi.reshape((nwalkers, nbasis, nocc))

    expH1 = numpy.random.randn(nbasis**2).reshape(nbasis, nbasis)

    phi_ref = numpy.einsum("mn,wni->wmi", expH1, phi)
    phi = propagate_one_body(phi, expH1)

    numpy.testing.assert_allclose(phi, phi_ref, atol=1e-10)


@pytest.mark.unit
def test_apply_exponential():
    nwalkers = 10
    nbasis = 25
    nocc = 10

    ndim = nwalkers * nbasis * nocc
    phi = numpy.random.randn(ndim) + 1.0j * numpy.random.randn(ndim)
    phi = phi.reshape((nwalkers, nbasis, nocc))

    phi_ref = numpy.zeros_like(phi)

    VHS = 0.005 * (
        numpy.random.randn(nwalkers * nbasis * nbasis)
        + 1.0j * numpy.random.randn(nwalkers * nbasis * nbasis)
    )
    VHS = VHS.reshape((nwalkers, nbasis, nbasis))
    for iw in range(nwalkers):
        VHS[iw] = VHS[iw] + VHS[iw].T.conj()
        expV = scipy.linalg.expm(VHS[iw])
        phi_ref[iw] = expV.dot(phi[iw])

    phi1 = numpy.copy(phi)

    for iw in range(nwalkers):
        phi1[iw] = apply_exponential(phi1[iw], VHS[iw], 6)

    phi2 = apply_exponential_batch(phi, VHS, 6)

    numpy.testing.assert_allclose(phi2, phi_ref, atol=1e-10)
    numpy.testing.assert_allclose(phi1, phi_ref, atol=1e-10)


if __name__ == "__main__":
    test_propagate_one_body()
    test_apply_exponential()
