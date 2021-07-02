import os
import unittest
import numpy
import pytest
from mpi4py import MPI
from pauxy.systems.generic import Generic
from pauxy.systems.utils import get_generic_integrals
from pauxy.utils.testing import generate_hamiltonian


@pytest.mark.unit
def test_real():
    numpy.random.seed(7)
    nmo = 17
    nelec = (4,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    assert sys.nup == 4
    assert sys.ndown == 3
    assert numpy.trace(h1e) == pytest.approx(9.38462274882365)


@pytest.mark.unit
def test_complex():
    numpy.random.seed(7)
    nmo = 17
    nelec = (5,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    assert sys.nup == 5
    assert sys.ndown == 3
    assert sys.nbasis == 17

@pytest.mark.unit
def test_write():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4,3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    sys.write_integrals()

@pytest.mark.unit
def test_read():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4,3)
    h1e_, chol_, enuc_, eri_ = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    from pauxy.utils.io import write_qmcpack_dense
    chol_ = chol_.reshape((-1,nmo*nmo)).T.copy()
    write_qmcpack_dense(h1e_, chol_, nelec, nmo,
                        enuc=enuc_, filename='hamil.h5',
                        real_chol=False)
    filename = 'hamil.h5'
    nup, ndown = nelec
    comm = None
    hcore, chol, h1e_mod, enuc = get_generic_integrals(filename,
                                                       comm=comm,
                                                       verbose=False)
    system = Generic(h1e=hcore, chol=chol, ecore=enuc,
                     h1e_mod=h1e_mod, nelec=nelec,
                     verbose=False)
    assert system.ecore == pytest.approx(0.4392816555570978)
    assert system.chol_vecs.shape == chol_.shape
    assert len(system.H1.shape) == 3
    assert numpy.linalg.norm(system.H1[0]-h1e_) == pytest.approx(0.0)
    assert numpy.linalg.norm(system.chol_vecs-chol_) == pytest.approx(0.0)

@pytest.mark.unit
def test_shmem():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4,3)
    comm = MPI.COMM_WORLD
    h1e_, chol_, enuc_, eri_ = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    from pauxy.utils.io import write_qmcpack_dense
    chol_ = chol_.reshape((-1,nmo*nmo)).T.copy()
    write_qmcpack_dense(h1e_, chol_, nelec, nmo,
                        enuc=enuc_, filename='hamil.h5',
                        real_chol=False)
    filename = 'hamil.h5'
    nup, ndown = nelec
    from pauxy.utils.mpi import get_shared_comm
    shared_comm = get_shared_comm(comm, verbose=True)
    hcore, chol, h1e_mod, enuc = get_generic_integrals(filename,
                                                       comm=get_shared_comm,
                                                       verbose=False)
    system = Generic(h1e=hcore, chol=chol, ecore=enuc,
                     h1e_mod=h1e_mod, nelec=nelec,
                     verbose=False)
    assert system.ecore == pytest.approx(0.4392816555570978)
    assert system.chol_vecs.shape == chol_.shape
    assert len(system.H1.shape) == 3
    assert numpy.linalg.norm(system.H1[0]-h1e_) == pytest.approx(0.0)
    assert numpy.linalg.norm(system.chol_vecs-chol_) == pytest.approx(0.0)

def teardown_module():
    cwd = os.getcwd()
    files = ['hamil.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
