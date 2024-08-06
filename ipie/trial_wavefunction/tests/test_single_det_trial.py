import numpy
import pytest

from ipie.config import MPI
from ipie.estimators.utils import gab_spin
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.hamiltonians.generic import Generic as HamGeneric
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


@pytest.mark.unit
def test_single_det_ghf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)

    _wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    psi0a = _wavefunction[1][0][:, :nalpha]
    psi0b = _wavefunction[1][0][:, nalpha:]
    psi0a, _ = numpy.linalg.qr(psi0a)
    psi0b, _ = numpy.linalg.qr(psi0b)
    wavefunction = numpy.hstack([psi0a, psi0b])

    trial_uhf = SingleDet(wavefunction, (nalpha, nbeta), nbasis)
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial_uhf.half_rotate(ham)
    trial_uhf.calculate_energy(sys, ham)
    energy_ref = trial_uhf.energy

    # No rotation is applied.
    psi0 = numpy.zeros((2 * nbasis, nalpha + nbeta), dtype=trial_uhf.psi0a.dtype)
    psi0[:nbasis, :nalpha] = trial_uhf.psi0a.copy()
    psi0[nbasis:, nalpha:] = trial_uhf.psi0b.copy()

    trial = SingleDetGHF(psi0, (nalpha, nbeta), nbasis)
    trial.calculate_energy(sys, ham)

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(_wavefunction[0])
    numpy.testing.assert_allclose(trial_uhf.psi0a, trial.psi0[:nbasis, :nalpha], atol=1e-10)
    numpy.testing.assert_allclose(trial_uhf.psi0b, trial.psi0[nbasis:, nalpha:], atol=1e-10)
    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)

    # Applying spin-axis rotation and checking if the energy changes
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
    trial = SingleDetGHF(psi0, (nalpha, nbeta), nbasis)
    trial.calculate_energy(sys, ham)

    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)


@pytest.mark.unit
def test_single_det_complex_ghf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)

    _wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    psi0a = _wavefunction[1][0][:, :nalpha]
    psi0b = _wavefunction[1][0][:, nalpha:]
    psi0a, _ = numpy.linalg.qr(psi0a)
    psi0b, _ = numpy.linalg.qr(psi0b)
    wavefunction = numpy.concatenate([psi0a, psi0b], axis=1)

    trial_uhf = SingleDet(wavefunction, (nalpha, nbeta), nbasis)
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial_uhf.half_rotate(ham)
    trial_uhf.calculate_energy(sys, ham)
    energy_ref = trial_uhf.energy

    # No rotation is applied.
    psi0 = numpy.zeros((2 * nbasis, nalpha + nbeta), dtype=trial_uhf.psi0a.dtype)
    psi0[:nbasis, :nalpha] = trial_uhf.psi0a.copy()
    psi0[nbasis:, nalpha:] = trial_uhf.psi0b.copy()

    trial = SingleDetGHF(psi0, (nalpha, nbeta), nbasis)
    trial.calculate_energy(sys, ham)

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(_wavefunction[0])
    numpy.testing.assert_allclose(trial_uhf.psi0a, trial.psi0[:nbasis, :nalpha], atol=1e-10)
    numpy.testing.assert_allclose(trial_uhf.psi0b, trial.psi0[nbasis:, nalpha:], atol=1e-10)
    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)

    # Applying spin-axis rotation and checking if the energy changes.
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
    trial = SingleDetGHF(psi0, (nalpha, nbeta), nbasis)
    trial.calculate_energy(sys, ham)

    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)


@pytest.mark.unit
def test_single_det_ghf_from_uhf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)

    _wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    psi0a = _wavefunction[1][0][:, :nalpha]
    psi0b = _wavefunction[1][0][:, nalpha:]
    psi0a, _ = numpy.linalg.qr(psi0a)
    psi0b, _ = numpy.linalg.qr(psi0b)
    wavefunction = numpy.concatenate([psi0a, psi0b], axis=1)

    trial_uhf = SingleDet(wavefunction, (nalpha, nbeta), nbasis)
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    trial_uhf.half_rotate(ham)
    trial_uhf.calculate_energy(sys, ham)
    energy_ref = trial_uhf.energy

    trial = SingleDetGHF(trial_uhf)
    trial.calculate_energy(sys, ham)

    numpy.testing.assert_allclose(trial_uhf.psi0a, trial.psi0[:nbasis, :nalpha], atol=1e-10)
    numpy.testing.assert_allclose(trial_uhf.psi0b, trial.psi0[nbasis:, nalpha:], atol=1e-10)
    numpy.testing.assert_almost_equal(energy_ref, trial.energy, decimal=10)


@pytest.mark.unit
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


# pytest cannot import `ueg` if we just run `pytest -v`.
@pytest.mark.skip()
def test_calculate_energy_complex_ueg():
    from ueg import UEG
    from pyscf import gto, scf, ao2mo

    # Generate UEG integrals.
    ueg_opts = {
        "nup": 7,
        "ndown": 7,
        "rs": 1.0,
        "ecut": 2.5,
        "thermal": False,
        "write_integrals": False,
    }

    ueg = UEG(ueg_opts)
    ueg.build()
    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nelec = (ueg.nup, ueg.ndown)
    nup, ndown = nelec

    h1 = ueg.H1[0]
    chol = 2.0 * ueg.chol_vecs.toarray()
    # ecore = ueg.ecore
    ecore = 0.0

    # -------------------------------------------------------------------------
    # Build trial wavefunction.
    # For pyscf.
    U = ueg.compute_real_transformation()
    h1_8 = U.T.conj() @ h1 @ U
    eri_8 = ueg.eri_8()  # 8-fold eri
    eri_8 = ao2mo.restore(8, eri_8, nbasis)

    mol = gto.M()
    mol.nelectron = numpy.sum(nelec)
    mol.spin = nup - ndown
    mol.max_memory = 60000  # MB
    mol.incore_anyway = True

    # PW guess.
    dm0a = numpy.zeros(nbasis)
    dm0b = numpy.zeros(nbasis)
    dm0a[:nup] = 1
    dm0b[:ndown] = 1
    dm0 = numpy.array([numpy.diag(dm0a), numpy.diag(dm0b)])

    mf = scf.UHF(mol)
    # mf.level_shift = 0.5
    mf.max_cycle = 5000
    mf.get_hcore = lambda *args: h1_8
    mf.get_ovlp = lambda *args: numpy.eye(nbasis)
    mf._eri = eri_8
    emf = mf.kernel(dm0)

    Ca, Cb = mf.mo_coeff
    psia = Ca[:, :nup]
    psib = Cb[:, :ndown]
    psi0 = numpy.zeros((nbasis, numpy.sum(nelec)), dtype=numpy.complex128)
    psi0[:, :nup] = psia
    psi0[:, nup:] = psib

    # -------------------------------------------------------------------------
    # 1. Build out system.
    system = Generic(nelec=nelec)

    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
        numpy.array([h1, h1], dtype=numpy.complex128),
        numpy.array(chol, dtype=numpy.complex128),
        ecore,
    )

    # 3. Build trial.
    trial = SingleDet(psi0, nelec, nbasis)

    # Need to hard-code complex wfns for now.
    trial.psi = trial.psi.astype(numpy.complex128)
    trial.psi0a = trial.psi0a.astype(numpy.complex128)
    trial.psi0b = trial.psi0b.astype(numpy.complex128)
    trial.G, trial.Ghalf = gab_spin(trial.psi, trial.psi, trial.nalpha, trial.nbeta)
    trial.half_rotate(hamiltonian)
    trial.calculate_energy(system, hamiltonian)

    # Check against RHF solutions of 10.1063/1.5109572
    numpy.testing.assert_allclose(numpy.around(trial.energy, 6), 13.603557)  # rs = 1, nbasis = 57
    numpy.testing.assert_allclose(trial.energy, emf)


if __name__ == "__main__":
    test_single_det()
    test_single_det_ghf()
    test_single_det_complex_ghf()
    test_single_det_ghf_from_uhf()
    test_single_det_all_up()
    test_calculate_energy_complex_ueg()
