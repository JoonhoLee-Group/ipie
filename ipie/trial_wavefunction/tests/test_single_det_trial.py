import numpy
import pytest

from ipie.config import MPI
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.testing import get_random_nomsd, get_random_sys_ham


def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        contents = ""
        for j in range(matrix.shape[1]):
            contents += f"{matrix[i, j]:5.3f} "
        print(contents)


@pytest.mark.unit
def test_single_det():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)
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


#@pytest.mark.unit
def test_single_det_ghf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)

    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    trial = SingleDet(
        wavefunction[1][0],
        (nalpha, nbeta),
        nbasis,
    )
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial.half_rotate(ham)
    trial.calculate_energy(sys, ham)

    energy_ref = trial.energy

    psi0 = numpy.zeros((2 * nbasis, nalpha + nbeta), dtype=trial.psi0a.dtype)

    # no rotation is applied
    trial.psi0a, _ = numpy.linalg.qr(trial.psi0a)
    trial.psi0b, _ = numpy.linalg.qr(trial.psi0b)
    psi0[:nbasis, :nalpha] = trial.psi0a.copy()
    psi0[nbasis:, nalpha:] = trial.psi0b.copy()

    trial = SingleDetGHF(
        psi0,
        (nalpha, nbeta),
        nbasis,
    )
    trial.calculate_energy(sys, ham)

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(wavefunction[0])

    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)

    # applying spin-axis rotation and checking if the energy changes
    theta = numpy.pi / 2.0
    phi = numpy.pi / 4.0

    Uspin = numpy.array(
        [
            [numpy.cos(theta / 2.0), -numpy.exp(1.0j * phi) * numpy.sin(theta / 2.0)],
            [numpy.exp(-1.0j * phi) * numpy.sin(theta / 2.0), numpy.cos(theta / 2.0)],
        ],
        dtype=numpy.complex128,
    )
    U = numpy.kron(Uspin, numpy.eye(trial.nbasis))

    psi0 = U.dot(psi0)
    trial = SingleDetGHF(
        psi0,
        (nalpha, nbeta),
        nbasis,
    )
    trial.calculate_energy(sys, ham)

    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)


def test_single_det_all_up():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 0)
    numpy.random.seed(7)
    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    trial = SingleDet(
        wavefunction[1][0],
        (nalpha, nbeta),
        nbasis,
    )

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(wavefunction[0])
    assert trial.G.shape == (2, nbasis, nbasis)
    assert trial.Ghalf[0].shape == (nalpha, nbasis)
    assert trial.Ghalf[1].shape == (nbeta, nbasis)

    trial.build()
    comm = MPI.COMM_WORLD
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial.half_rotate(ham, comm=comm)
    assert trial._rchola.shape == (naux, nbasis * nalpha)
    assert trial._rcholb.shape == (naux, nbasis * nbeta)
    assert trial._rH1a.shape == (nalpha, nbasis)
    assert trial._rH1b.shape == (nbeta, nbasis)


if __name__ == "__main__":
    test_single_det()
    #test_single_det_ghf()
    test_single_det_all_up()
