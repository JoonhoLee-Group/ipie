"""Hubbard Hamiltonian with a generic one-body hopping matrix and uniform U."""

import numpy

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.io import fcidump_header


def construct_h1e_mod(h1e, U, h1e_mod, symmetric=False):
    """Build the shifted one-body Hamiltonian used in propagation."""
    if symmetric:
        h1e_mod[:] = h1e
    else:
        v0 = 0.5 * U * numpy.eye(h1e.shape[-1], dtype=h1e.dtype)
        h1e_mod[0] = h1e[0] - v0
        h1e_mod[1] = h1e[1] - v0


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
    symmetric : bool, optional
        If ``True``, keep the symmetric Hubbard form without the one-body shift.
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
        """Dump one- and two-electron integrals in FCIDUMP format."""
        header = fcidump_header(nup + ndown, self.nbasis, nup - ndown)
        is_complex = numpy.iscomplexobj(self.T)

        for i in range(1, self.nbasis + 1):
            if is_complex:
                fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                header += fmt.format(self.U.real, self.U.imag, i, i, i, i)
            else:
                fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                header += fmt.format(self.U, i, i, i, i)

        for i in range(self.nbasis):
            for j in range(i + 1, self.nbasis):
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
