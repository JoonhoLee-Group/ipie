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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

import numpy

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp


def construct_h1e_mod_isdf(MPQ, cgto, h1e, h1e_mod):
    cgto_PQ = cgto @ cgto.T.conj()
    cgto_M = MPQ * cgto_PQ
    v0 = 0.5 * cgto.conj().T @ cgto_M @ cgto
    h1e_mod[0, :, :] = h1e[0] - v0
    h1e_mod[1, :, :] = h1e[1] - v0


class GenericRealISDF(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 8-fold real symmetric integrals.
    The electron repulsion integrals are approximated by Interpolative Separable Density Fitting (ISDF).
    """

    def __init__(
        self,
        h1e,
        MPQ,
        cholM,
        cgto,
        ecore=0.0,
        verbose=False,
        halfrot_cgto=None,
        halfrot_M=None,
        h1e_mod=None,
    ):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 3  # shape = nspin, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.MPQ = numpy.array(MPQ, dtype=numpy.float64)
        self.cholM = numpy.array(cholM, dtype=numpy.float64)  # [P, gamma], M = LL^\dagger
        self.nchol = self.cholM.shape[-1]
        # here we don't have spin indices for cgto because we use OAO basis for UHF cases to avoid extra storage
        self.cgto = numpy.array(cgto, dtype=numpy.float64)  # [P, p]
        if halfrot_cgto is not None:
            self.halfrot_cgtoa, self.halfrot_cgtob, self.halfrot_cgto = (
                halfrot_cgto  # [\tilde{P}, i(a)], [\tilde{P}, i(b)], [\tilde{P}, p]
            )
        else:
            self.halfrot_cgtoa = None
            self.halfrot_cgtob = None
            self.halfrot_cgto = self.cgto
        if halfrot_M is not None:
            self.halfrot_M = halfrot_M
        else:
            self.halfrot_M = None

        self.nisdf = self.cgto.shape[0]
        self.nisdf_halfrot = self.halfrot_cgto.shape[0]
        self.nfields = self.nchol
        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        if h1e_mod is not None:
            self.h1e_mod = xp.array(h1e_mod)
        else:
            h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
            construct_h1e_mod_isdf(self.MPQ, self.cgto, self.H1, h1e_mod)
            self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = 2 * self.cholM.nbytes / (1024.0**3) + 2 * self.cgto.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by ISDF vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptISDF object.")


class GenericComplexISDF(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals.
    The electron repulsion integrals are approximated by Interpolative Separable Density Fitting (ISDF).
    """

    def __init__(
        self,
        h1e,
        MPQ,
        cholM,
        cgto,
        ecore=0.0,
        verbose=False,
        halfrot_cgto=None,
        halfrot_M=None,
        h1e_mod=None,
    ):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 3  # shape = nspin, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.MPQ = numpy.array(MPQ, dtype=numpy.complex128)
        self.RcholM = numpy.real(cholM)  # [P, gamma], M = LL^\dagger
        self.IcholM = numpy.imag(cholM)  # [P, gamma], M = LL^\dagger
        self.nchol = self.cholM.shape[-1]
        # here we don't have spin indices for cgto because we use OAO basis for UHF cases to avoid extra storage
        self.cgto = numpy.array(cgto, dtype=numpy.complex128)  # [P, p]
        if halfrot_cgto is not None:
            self.halfrot_cgtoa, self.halfrot_cgtob, self.halfrot_cgto = (
                halfrot_cgto  # [\tilde{P}, i(a)], [\tilde{P}, i(b)], [\tilde{P}, p]
            )
        else:
            self.halfrot_cgtoa = None
            self.halfrot_cgtob = None
            self.halfrot_cgto = self.cgto
        if halfrot_M is not None:
            self.halfrot_M = halfrot_M
        else:
            self.halfrot_M = None

        self.nisdf = self.cgto.shape[0]
        self.nisdf_halfrot = self.halfrot_cgto.shape[0]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        if h1e_mod is not None:
            self.h1e_mod = xp.array(h1e_mod)
        else:
            h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
            construct_h1e_mod_isdf(self.MPQ, self.cgto, self.H1, h1e_mod)
            self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = 2 * self.cholM.nbytes / (1024.0**3) + 2 * self.cgto.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by ISDF vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptISDF object.")
