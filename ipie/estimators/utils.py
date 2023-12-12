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

import h5py
import numpy
import scipy


class H5EstimatorHelper(object):
    """Helper class for pushing data to hdf5 dataset of fixed length.

    Parameters
    ----------
    h5f : :class:`h5py.File`
        Output file object.
    name : string
        Dataset name.
    shape : tuple
        Shape of output data.
    dtype : type
        Output data type.

    Attributes
    ----------
    store : :class:`h5py.File.DataSet`
        Dataset object.
    index : int
        Counter for incrementing data.
    """

    def __init__(self, filename, base, chunk_size=1, shape=(1,)):
        # self.store = h5f.create_dataset(name, shape, dtype=dtype)
        self.filename = filename
        self.base = base
        self.index = 0
        self.chunk_index = 0
        self.nzero = 9
        self.chunk_size = chunk_size
        self.shape = (chunk_size,) + shape

    def push(self, data, name):
        """Push data to dataset.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Data to push.
        """
        ix = str(self.index)
        # To ensure string indices are sorted properly.
        padded = "0" * (self.nzero - len(ix)) + ix
        dset = self.base + "/" + name + "/" + padded
        with h5py.File(self.filename, "a") as fh5:
            fh5[dset] = data

    def push_to_chunk(self, data, name):
        """Push data to dataset.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            Data to push.
        """
        ix = str(self.index // self.chunk_size)
        # To ensure string indices are sorted properly.
        padded = "0" * (self.nzero - len(ix)) + ix
        dset = self.base + "/" + name + "/" + padded
        with h5py.File(self.filename, "a") as fh5:
            if dset in fh5:
                fh5[dset][self.chunk_index] = data
                fh5[self.base + f"/max_block/{ix}"][()] = self.chunk_index
            else:
                fh5[dset] = numpy.zeros(self.shape, dtype=numpy.complex128)
                fh5[dset][self.chunk_index] = data
                fh5[self.base + f"/max_block/{ix}"] = self.chunk_index

    def increment(self):
        self.index = self.index + 1
        self.chunk_index = (self.chunk_index + 1) % self.chunk_size

    def reset(self):
        self.index = 0


def gab_mod(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    O = numpy.dot(B.T, A.conj())
    GHalf = numpy.dot(scipy.linalg.inv(O), B.T)
    G = numpy.dot(A.conj(), GHalf)
    return (G, GHalf)


def gab_spin(A, B, na, nb):
    GA, GAH = gab_mod(A[:, :na], B[:, :na])
    if nb > 0:
        GB, GBH = gab_mod(A[:, na:], B[:, na:])
    else:
        GB = numpy.zeros(GA.shape, dtype=GA.dtype)
        GBH = numpy.zeros((0, GAH.shape[1]), dtype=GAH.dtype)
    return numpy.array([GA, GB]), [GAH, GBH]
