
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
                fh5[self.base+f'/max_block/{ix}'][()] = self.chunk_index
            else:
                fh5[dset] = numpy.zeros(self.shape, dtype=numpy.complex128)
                fh5[dset][self.chunk_index] = data
                fh5[self.base+f'/max_block/{ix}'] = self.chunk_index

    def increment(self):
        self.index = self.index + 1
        self.chunk_index = (self.chunk_index + 1) % self.chunk_size

    def reset(self):
        self.index = 0
