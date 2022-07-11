"""Hubbard model specific classes and methods"""

import cmath
import math
from math import cos, pi, sqrt

import numpy
import scipy.linalg

from ipie.utils.io import fcidump_header


class HubbardHolstein(object):
    """HubbardHolstein model system class.

    1 and 2 case with nearest neighbour hopping.

    Parameters
    ----------
    inputs : dict
        dictionary of system input options.

    Attributes
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    ne : int
        Number of electrons.
    t : float
        Hopping parameter.
    U : float
        Hubbard U interaction strength.
    g : float
        Holstein coupling strength
    lmbda : float
        Polaron formation energy
    w0 : float
        Frequency w0
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    nbasis : int
        Number of single-particle basis functions.
    T : numpy.array
        Hopping matrix
    """

    def __init__(self, inputs, verbose=False):
        if verbose:
            print("# Parsing input options.")
        self.nup = inputs.get("nup")
        self.ndown = inputs.get("ndown")
        self.ne = self.nup + self.ndown
        self.nelec = (self.nup, self.ndown)

        self.control_variate = False

        self.t = inputs.get("t", 1.0)
        self.lmbda = inputs.get("lambda", 1.0)
        self.w0 = inputs.get("w0", 1.0)
        self.m = inputs.get("m", 1.0 / self.w0)  # mass

        self.U = inputs["U"]
        self.nx = inputs["nx"]
        self.ny = inputs["ny"]
        self.ktwist = numpy.array(inputs.get("ktwist"))
        self.symmetric = inputs.get("symmetric", False)

        self.ypbc = inputs.get("ypbc", True)
        self.xpbc = inputs.get("xpbc", True)

        self.control_variate = False

        if self.symmetric:
            # An unusual convention for the sign of the chemical potential is
            # used in Phys. Rev. B 99, 045108 (2018)
            # Symmetric uses the symmetric form of the hubbard model and will
            # also change the sign of the chemical potential in the density
            # matrix.
            self._alt_convention = True
        else:
            self._alt_convention = False
        self.nbasis = self.nx * self.ny

        # This depends on the dimension of the system hard-coded to do 1D
        d = 2
        if self.nx == 1 or self.ny == 1:  # 1d
            d = 1

        self.g = inputs.get("g", None)

        if self.g == None:
            # This is assuming self.m = 1 / self.w0
            # to include mass see 10.1103/PhysRevLett.97.056402
            self.g = sqrt(float(d) * 2.0 * self.lmbda * self.t * self.w0)

        if verbose:
            print("# d = {}".format(d))
            print("# nx, ny = {},{}".format(self.nx, self.ny))
            print("# nbasis = {}".format(self.nbasis))
            print("# t, U = {}, {}".format(self.t, self.U))
            print(
                "# m, w0, g, lambda = {}, {}, {}, {}".format(
                    self.m, self.w0, self.g, self.lmbda
                )
            )

        self.lang_firsov = inputs.get("lang_firsov", False)
        self.gamma_lf = 0.0
        if self.lang_firsov:
            self.gamma_lf = self.g * numpy.sqrt(2.0 / (self.m * self.w0**3))
            self.Ueff = (
                self.U
                + self.gamma_lf**2 * self.m * self.w0**2
                - 2.0 * self.g * self.gamma_lf * numpy.sqrt(2.0 * self.m * self.w0)
            )
            print("# gamma_lf = {}".format(self.gamma_lf))
            print("# Ueff = {}".format(self.Ueff))

        self.nactive = self.nbasis
        self.nfv = 0
        self.ncore = 0
        (self.kpoints, self.kc, self.eks) = kpoints(self.t, self.nx, self.ny)
        self.pinning = inputs.get("pinning_fields", False)
        self._opt = True
        if verbose:
            print("# Setting up one-body operator.")
        if self.pinning:
            if verbose:
                print("# Using pinning field.")
            self.T = kinetic_pinning_alt(self.t, self.nbasis, self.nx, self.ny)
        else:
            self.T = kinetic(
                self.t,
                self.nbasis,
                self.nx,
                self.ny,
                self.ktwist,
                xpbc=self.xpbc,
                ypbc=self.ypbc,
            )
        self.H1 = self.T
        self.Text = scipy.linalg.block_diag(self.T[0], self.T[1])
        self.P = transform_matrix(self.nbasis, self.kpoints, self.kc, self.nx, self.ny)
        self.mu = inputs.get("mu", None)
        # For interface consistency.
        self.ecore = 0.0
        # Number of field configurations per walker.
        self.nfields = self.nbasis
        self.name = "HubbardHolstein"
        if verbose:
            print("# Finished setting up Hubbard-Holstein system object.")
        # "Volume" to define density.
        self.vol = self.nx * self.ny
        self.construct_h1e_mod()

    def fcidump(self, to_string=False):
        """Dump 1- and 2-electron integrals to file.

        Parameters
        ----------
        to_string : bool
            Return fcidump as string. Default print to stdout.
        """
        header = fcidump_header(self.ne, self.nbasis, self.nup - self.ndown)
        for i in range(1, self.nbasis + 1):
            if self.T.dtype == complex:
                fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                line = fmt.format(self.U.real, self.U.imag, i, i, i, i)
            else:
                fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                line = fmt.format(self.U, i, i, i, i)
            header += line
        for i in range(0, self.nbasis):
            for j in range(i + 1, self.nbasis):
                integral = self.T[0][i, j]
                if abs(integral) > 1e-8:
                    if self.T.dtype == complex:
                        fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        line = fmt.format(
                            integral.real, integral.imag, i + 1, j + 1, 0, 0
                        )
                    else:
                        fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
                        line = fmt.format(integral, i + 1, j + 1, 0, 0)
                    header += line
        if self.T.dtype == complex:
            fmt = "({: 10.8e}, {: 10.8e}) {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0, 0)
        else:
            fmt = "{: 10.8e} {:>3d} {:>3d} {:>3d} {:>3d}\n"
            header += fmt.format(0, 0, 0, 0, 0)
        if to_string:
            print(header)
        else:
            return header

    def hijkl(self, i, j, k, l):
        if i == k and j == l and i == j:
            return self.U
        else:
            return 0.0

    def construct_h1e_mod(self):
        # Subtract one-body bit following reordering of 2-body operators.
        # Eqn (17) of [Motta17]_
        if not self.symmetric:
            v0 = 0.5 * self.U * numpy.eye(self.nbasis)
            self.h1e_mod = numpy.array([self.H1[0] - v0, self.H1[1] - v0])
        else:
            self.h1e_mod = self.H1


def transform_matrix(nbasis, kpoints, kc, nx, ny):
    U = numpy.zeros(shape=(nbasis, nbasis), dtype=complex)
    for (i, k_i) in enumerate(kpoints):
        for j in range(0, nbasis):
            r_j = decode_basis(nx, ny, j)
            U[i, j] = numpy.exp(1j * numpy.dot(kc * k_i, r_j))

    return U


def kinetic(t, nbasis, nx, ny, ks, xpbc=True, ypbc=True):
    """Kinetic part of the Hamiltonian in our one-electron basis.

    Parameters
    ----------
    t : float
        Hopping parameter
    nbasis : int
        Number of one-electron basis functions.
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.

    Returns
    -------
    T : numpy.array
        Hopping Hamiltonian matrix.
    """

    if ks.all() is None:
        T = numpy.zeros((nbasis, nbasis), dtype=float)
    else:
        T = numpy.zeros((nbasis, nbasis), dtype=complex)

    for i in range(0, nbasis):
        xy1 = decode_basis(nx, ny, i)
        for j in range(i + 1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1 - xy2)
            if sum(dij) == 1:
                T[i, j] = -t
            # Take care of periodic boundary conditions
            # there should be a less stupid way of doing this.
            if ny == 1 and dij == [nx - 1] and xpbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j * numpy.dot(cmath.pi * ks, [1]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase
            elif (dij == [nx - 1, 0]).all() and xpbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j * numpy.dot(cmath.pi * ks, [1, 0]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase
            elif (dij == [0, ny - 1]).all() and ypbc:
                if ks.all() is not None:
                    phase = cmath.exp(1j * numpy.dot(cmath.pi * ks, [0, 1]))
                else:
                    phase = 1.0
                T[i, j] += -t * phase

    # This only works because the diagonal of T is zero.
    return numpy.array([T + T.conj().T, T + T.conj().T])


def kinetic_pinning(t, nbasis, nx, ny):
    r"""Kinetic part of the Hamiltonian in our one-electron basis.

    Adds pinning fields as outlined in [Qin16]_. This forces periodic boundary
    conditions along x and open boundary conditions along y. Pinning fields are
    applied in the y direction as:

        .. math::
            \nu_{i\uparrow} = -\nu_{i\downarrow} = (-1)^{i_x+i_y}\nu_0,

    for :math:`i_y=1,L_y` and :math:`\nu_0=t/4`.

    Parameters
    ----------
    t : float
        Hopping parameter
    nbasis : int
        Number of one-electron basis functions.
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.

    Returns
    -------
    T : numpy.array
        Hopping Hamiltonian matrix.
    """

    Tup = numpy.zeros((nbasis, nbasis))
    Tdown = numpy.zeros((nbasis, nbasis))
    nu0 = 0.25 * t

    for i in range(0, nbasis):
        # pinning field along y.
        xy1 = decode_basis(nx, ny, i)
        if xy1[1] == 0 or xy1[1] == ny - 1:
            Tup[i, i] += (-1.0) ** (xy1[0] + xy1[1]) * nu0
            Tdown[i, i] += (-1.0) ** (xy1[0] + xy1[1] + 1) * nu0
        for j in range(i + 1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1 - xy2)
            if sum(dij) == 1:
                Tup[i, j] = Tdown[i, j] = -t
            # periodic bcs in x.
            if (dij == [nx - 1, 0]).all():
                Tup[i, j] += -t
                Tdown[i, j] += -t

    return numpy.array([Tup + numpy.triu(Tup, 1).T, Tdown + numpy.triu(Tdown, 1).T])


def kinetic_pinning_alt(t, nbasis, nx, ny):
    r"""Kinetic part of the Hamiltonian in our one-electron basis.

    Adds pinning fields as outlined in [Qin16]_. This forces periodic boundary
    conditions along x and open boundary conditions along y. Pinning fields are
    applied in the y direction as:

        .. math::
            \nu_{i\uparrow} = -\nu_{i\downarrow} = (-1)^{i_x+i_y}\nu_0,

    for :math:`i_y=1,L_y` and :math:`\nu_0=t/4`.

    Parameters
    ----------
    t : float
        Hopping parameter
    nbasis : int
        Number of one-electron basis functions.
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.

    Returns
    -------
    T : numpy.array
        Hopping Hamiltonian matrix.
    """

    Tup = numpy.zeros((nbasis, nbasis))
    Tdown = numpy.zeros((nbasis, nbasis))
    h = 0.1 * t

    for i in range(0, nbasis):
        # pinning field along y direction when i_x = 0.
        xy1 = decode_basis(nx, ny, i)
        if xy1[0] == 0:
            Tup[i, i] += (-1.0) ** (xy1[1]) * h
            Tdown[i, i] += (-1.0) ** (xy1[1] + 1) * h
        for j in range(i + 1, nbasis):
            xy2 = decode_basis(nx, ny, j)
            dij = abs(xy1 - xy2)
            if sum(dij) == 1:
                Tup[i, j] = Tdown[i, j] = -t
            # periodic bcs in y.
            if (dij == [0, ny - 1]).all():
                Tup[i, j] += -t
                Tdown[i, j] += -t

    return numpy.array([Tup + numpy.triu(Tup, 1).T, Tdown + numpy.triu(Tdown, 1).T])


def decode_basis(nx, ny, i):
    """Return cartesian lattice coordinates from basis index.

    Consider a 3x3 lattice then we index lattice sites like::

        (0,2) (1,2) (2,2)       6 7 8
        (0,1) (1,1) (2,1)  ->   3 4 5
        (0,0) (1,0) (2,0)       0 1 2

    i.e., i = i_x + n_x * i_y, and i_x = i%n_x, i_y = i//nx.

    Parameters
    ----------
    nx : int
        Number of x lattice sites.
    ny : int
        Number of y lattice sites.
    i : int
        Basis index (same for up and down spins).
    """
    if ny == 1:
        return numpy.array([i % nx])
    else:
        return numpy.array([i % nx, i // nx])


def encode_basis(i, j, nx):
    """Encode 2d index to one dimensional index.

    See decode basis for layout.

    Parameters
    ----------
    i : int
        x coordinate.
    j : int
        y coordinate
    nx : int
        Number of x lattice sites.

    Returns
    -------
    ix : int
        basis index.
    """
    return i + j * nx


def _super_matrix(U, nbasis):
    """Construct super-matrix from v_{ijkl}"""


def kpoints(t, nx, ny):
    """Construct kpoints for system.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    nx : int
        Number of x lattice sites.
    nx : int
        Number of y lattice sites.

    Returns
    -------
    kp : numpy array
        System kpoints Note these are not sorted according to eigenvalue energy
        but rather so as to conform with numpys default kpoint indexing for FFTs.
    kfac : float
        Kpoint scaling factor (2pi/L).
    eigs : numpy array
        Single particle eigenvalues associated with kp.
    """
    kp = []
    eigs = []
    if ny == 1:
        kfac = numpy.array([2.0 * pi / nx])
        for n in range(0, nx):
            kp.append(numpy.array([n]))
            eigs.append(ek(t, n, kfac, ny))
    else:
        kfac = numpy.array([2.0 * pi / nx, 2.0 * pi / ny])
        for n in range(0, nx):
            for m in range(0, ny):
                k = numpy.array([n, m])
                kp.append(k)
                eigs.append(ek(t, k, kfac, ny))

    eigs = numpy.array(eigs)
    kp = numpy.array(kp)
    return (kp, kfac, eigs)


def ek(t, k, kc, ny):
    """Calculate single-particle energies.

    Parameters
    ----------
    t : float
        Hopping amplitude.
    k : numpy array
        Kpoint.
    kc : float
        Scaling factor.
    ny : int
        Number of y lattice points.
    """
    if ny == 1:
        e = -2.0 * t * cos(kc * k)
    else:
        e = -2.0 * t * (cos(kc[0] * k[0]) + cos(kc[1] * k[1]))

    return e


def get_strip(cfunc, cfunc_err, ix, nx, ny, stag=False):
    iy = [i for i in range(ny)]
    idx = [encode_basis(ix, i, nx) for i in iy]
    if stag:
        c = [((-1) ** (ix + i)) * cfunc[ib] for (i, ib) in zip(iy, idx)]
    else:
        c = [cfunc[ib] for ib in idx]
    cerr = [cfunc_err[ib] for ib in idx]
    return c, cerr
