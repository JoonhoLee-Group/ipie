# Copyright 2026 The ipie Developers. All Rights Reserved.
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

"""Hubbard Hamiltonian with a generic one-body hopping matrix and uniform U."""

import numpy

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.io import fcidump_header


class Hubbard(GenericBase):
    """Hamiltonian for a Hubbard interaction with site-independent U.

    Parameters
    ----------
    h1e : numpy.ndarray
        Spin-resolved one-body Hamiltonian with shape ``(2, nbasis, nbasis)``.
    U : float or complex
        Uniform on-site interaction strength.
    mu : float, optional
        Chemical potential retained for interface compatibility.
    ecore : float, optional
        Constant energy shift.
    verbose : bool, optional
        Print basic setup information.
    """

    def __init__(self, h1e, U, mu=None, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        super().__init__(h1e, ecore=ecore, verbose=verbose)

        self.mixed_precision = False
        self.chunked = False

        self.U = U
        self.mu = mu

        self.T = self.H1
        self.nfields = self.nbasis

        if verbose:
            print("# Number of orbitals: %d" % self.nbasis)
            print("# Number of fields: %d" % self.nfields)
            print("# Finished setting up Hubbard object.")

    def fcidump(self, nup, ndown, to_string=False):
        """Dump one- and two-electron integrals in FCIDUMP format (spin-independent hopping)."""
        assert numpy.allclose(self.T[0], self.T[1]), "FCIDUMP needs spin-independent hopping."
        header = fcidump_header(nup + ndown, self.nbasis, nup - ndown)
        is_complex = numpy.iscomplexobj(self.T)

        for i in range(1, self.nbasis + 1):
            if is_complex:
                fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                header += fmt.format(self.U.real, self.U.imag, i, i, i, i)
            else:
                fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                header += fmt.format(self.U, i, i, i, i)

        # Upper triangle including the diagonal (on-site energies); the reader fills h_ji = h_ij^*.
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                integral = self.T[0, i, j]
                if abs(integral) > 1e-8:
                    if is_complex:
                        fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        header += fmt.format(integral.real, integral.imag, i + 1, j + 1, 0, 0)
                    else:
                        fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        header += fmt.format(integral, i + 1, j + 1, 0, 0)

        if is_complex:
            fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0, 0)
        else:
            fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0)

        if to_string:
            return header

        print(header)

    def hijkl(self, i, j, k, l):
        """Return the Hubbard two-electron integral (ik|jl)."""
        if i == j == k == l:
            return self.U
        return 0.0


class HubbardSpinOrbital(Hubbard):
    """Hubbard interaction ``U sum_i n_{i up} n_{i down}`` with a spin-orbital one-body matrix.

    The one-body matrix acts on the ``2 * nbasis`` spin orbitals (all up orbitals first, then all
    down orbitals) and may couple the two spins, e.g. through a spin-flip field. Such a term conserves
    only the total particle number, so this Hamiltonian is for GHF trials and walkers.

    Parameters
    ----------
    h1e : numpy.ndarray
        Hermitian one-body matrix with shape ``(2 * nbasis, 2 * nbasis)``.
    U : float
        Uniform on-site interaction. ``U < 0`` is propagated with the real charge-channel Hirsch
        decoupling.
    ecore : float, optional
        Constant energy shift.
    verbose : bool, optional
        Print basic setup information.
    """

    def __init__(self, h1e, U, ecore=0.0, verbose=False):
        assert h1e.ndim == 2 and h1e.shape[0] == h1e.shape[1] and h1e.shape[0] % 2 == 0
        assert numpy.allclose(h1e, h1e.conj().T)
        GenericBase.__init__(self, h1e, ecore=ecore, verbose=verbose)
        self.nbasis = h1e.shape[0] // 2

        self.mixed_precision = False
        self.chunked = False

        self.U = U
        self.nfields = self.nbasis

        if verbose:
            print(f"# Number of orbitals: {self.nbasis} (spin-orbital one-body matrix)")
            print(f"# Number of fields: {self.nfields}")
            print("# Finished setting up HubbardSpinOrbital object.")
