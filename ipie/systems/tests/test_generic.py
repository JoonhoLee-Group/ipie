
# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

import os
import unittest

import numpy
import pytest
from mpi4py import MPI

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.utils import get_generic_integrals
from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian


@pytest.mark.unit
def test_real():
    numpy.random.seed(7)
    nmo = 17
    nelec = (4, 3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )
    assert sys.nup == 4
    assert sys.ndown == 3
    assert numpy.trace(h1e) == pytest.approx(9.38462274882365)


@pytest.mark.unit
def test_complex():
    numpy.random.seed(7)
    nmo = 17
    nelec = (5, 3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )
    assert sys.nup == 5
    assert sys.ndown == 3
    assert ham.nbasis == 17


@pytest.mark.unit
def test_write():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )
    ham.write_integrals(nelec, filename="hamil.test_write.h5")


@pytest.mark.unit
def test_read():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e_, chol_, enuc_, eri_ = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    from ipie.utils.io import write_qmcpack_dense

    chol_ = chol_.reshape((-1, nmo * nmo)).T.copy()
    filename = "hamil.test_read.h5"
    write_qmcpack_dense(
        h1e_, chol_, nelec, nmo, enuc=enuc_, filename=filename, real_chol=False
    )
    nup, ndown = nelec
    comm = None
    hcore, chol, h1e_mod, enuc = get_generic_integrals(
        filename, comm=comm, verbose=False
    )
    sys = Generic(nelec=nelec)
    ham = HamGeneric(h1e=hcore, chol=chol, ecore=enuc)
    assert ham.ecore == pytest.approx(0.4392816555570978)
    assert ham.chol_vecs.shape == chol_.shape  # now two are transposed
    assert len(ham.H1.shape) == 3
    assert numpy.linalg.norm(ham.H1[0] - h1e_) == pytest.approx(0.0)
    assert numpy.linalg.norm(ham.chol_vecs - chol_) == pytest.approx(
        0.0
    )  # now two are transposed


@pytest.mark.unit
def test_shmem():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    comm = MPI.COMM_WORLD
    h1e_, chol_, enuc_, eri_ = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    from ipie.utils.io import write_qmcpack_dense

    chol_ = chol_.reshape((-1, nmo * nmo)).T.copy()
    filename = "hamil.test_shmem.h5"
    write_qmcpack_dense(
        h1e_, chol_, nelec, nmo, enuc=enuc_, filename=filename, real_chol=False
    )
    filename = "hamil.test_shmem.h5"
    nup, ndown = nelec
    from ipie.utils.mpi import get_shared_comm

    shared_comm = get_shared_comm(comm, verbose=True)
    hcore, chol, h1e_mod, enuc = get_generic_integrals(
        filename, comm=get_shared_comm, verbose=False
    )
    # system = Generic(h1e=hcore, chol=chol, ecore=enuc,
    #                  h1e_mod=h1e_mod, nelec=nelec,
    #                  verbose=False)
    # print("hcore.shape = ", hcore.shape)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(h1e=hcore, h1e_mod=h1e_mod, chol=chol.copy(), ecore=enuc)

    assert ham.ecore == pytest.approx(0.4392816555570978)
    assert ham.chol_vecs.shape == chol_.shape  # now two are transposed
    assert len(ham.H1.shape) == 3
    assert numpy.linalg.norm(ham.H1[0] - h1e_) == pytest.approx(0.0)
    assert numpy.linalg.norm(ham.chol_vecs - chol_) == pytest.approx(
        0.0
    )  # now two are transposed


def teardown_module():
    cwd = os.getcwd()
    files = ["hamil.test_read.h5", "hamil.test_shmem.h5", "hamil.test_write.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass


if __name__ == "__main__":
    test_write()
    test_read()
    test_shmem()
