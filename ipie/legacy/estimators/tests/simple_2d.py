import itertools
import os
import unittest

import numpy
from scipy.fftpack.helper import next_fast_len

from ipie.legacy.estimators.greens_function import gab_mod
from ipie.utils.testing import get_random_wavefunction


def create_grid(x):
    nx = len(x)
    # kval = numpy.array(list(itertools.product(*[gx,gy,gz])), dtype=numpy.int32)
    grid = numpy.zeros((nx, 3), dtype=numpy.int32)
    for ix in range(nx):
        grid[ix, :] = [x[ix]]
    return grid


def fill_up_range(nmesh):
    a = numpy.zeros(nmesh)
    n = nmesh // 2
    a = numpy.linspace(-n, n, num=nmesh, dtype=numpy.int32)
    return a


def generate_fft_grid(mesh):
    gx = fill_up_range(mesh[0])
    gy = fill_up_range(mesh[0])
    kval = numpy.array(list(itertools.product(*[gx, gy])), dtype=numpy.int32)
    spval = 0.5 * numpy.array([numpy.dot(g, g) for g in kval])

    return kval, spval


def lookup(g, basis):
    for i, k in enumerate(basis):
        if numpy.dot(g - k, g - k) == 0:
            return i
    return None


# Gives same results as scipy.signal.convolve
def convolve(f, g, mesh):
    f_ = f.reshape(*mesh)
    g_ = g.reshape(*mesh)
    shape = numpy.maximum(f_.shape, g_.shape)
    min_shape = numpy.array(f_.shape) + numpy.array(g_.shape) - 1
    fshape = [next_fast_len(d) for d in min_shape]
    fslice = tuple([slice(sz) for sz in shape])
    fq = numpy.fft.fftn(
        numpy.fft.ifftn(f_, s=fshape) * numpy.fft.ifftn(g_, s=fshape)
    ).copy()
    return fq.ravel()


def test_fft_kpq(nalpha):
    # Create regular grid.
    nmax = 2
    mesh = [2 * nmax + 1] * 2
    grid, eigs = generate_fft_grid(mesh)

    qmax = 2 * nmax
    qmesh = [2 * qmax + 1] * 2
    qgrid, qeigs = generate_fft_grid(qmesh)

    # Create wavefunction
    nbasis = len(grid)

    numpy.random.seed(7)

    psi = get_random_wavefunction((nalpha, nalpha), nbasis)
    I = get_random_wavefunction((nalpha, nalpha), nbasis)
    trial = I[:, :nalpha].conj()

    G, Gh = gab_mod(trial, psi[:, :nalpha])
    nqgrid = numpy.prod(qmesh)

    # Check by direct convolution f(q) = \sum_G Psi[G+Q] Gh[G].
    fq_direct = numpy.zeros(nqgrid, dtype=numpy.complex128)
    for i, q in enumerate(qgrid):
        for j, k in enumerate(grid):
            kpq = k + q
            ikpq = lookup(kpq, grid)
            if ikpq is not None:
                fq_direct[i] += trial[ikpq, 0] * Gh[0, j]

    trial_grid = numpy.flip(trial[:, 0]).reshape(mesh)
    Gh_grid = Gh[0, :].reshape(mesh)

    # Check by fft convolve
    # Compare to fq
    fq_conv = numpy.zeros(nqgrid, dtype=numpy.complex128)
    fq_conv += nqgrid * convolve(Gh_grid, trial_grid, mesh)
    fq_conv = numpy.flip(fq_conv)

    import scipy.signal

    fq_conv_sc = numpy.flip(scipy.signal.fftconvolve(Gh_grid, trial_grid).ravel())

    import matplotlib.pyplot as pl

    pl.plot(fq_conv, label="fft")
    pl.plot(fq_conv_sc, label="fft_scipy")
    pl.plot(fq_direct, label="direct")
    # pl.plot(fq, label='from_gf')
    pl.legend()
    pl.show()
    # print(Gtrace, Gtrace_direct, fq[1])


def test_fft_kmq(nalpha):
    # Create regular grid.
    nmax = 2
    mesh = [2 * nmax + 1] * 2
    grid, eigs = generate_fft_grid(mesh)

    qmax = 2 * nmax
    qmesh = [2 * qmax + 1] * 2
    qgrid, qeigs = generate_fft_grid(qmesh)

    # Create wavefunction
    nbasis = len(grid)

    numpy.random.seed(7)

    psi = get_random_wavefunction((nalpha, nalpha), nbasis)
    I = get_random_wavefunction((nalpha, nalpha), nbasis)
    trial = I[:, :nalpha].conj()

    G, Gh = gab_mod(trial, psi[:, :nalpha])
    nqgrid = numpy.prod(qmesh)

    # Check by direct convolution f(q) = \sum_G Psi[G+Q] Gh[G].
    fq_direct = numpy.zeros(nqgrid, dtype=numpy.complex128)
    for i, q in enumerate(qgrid):
        for j, k in enumerate(grid):
            kmq = k - q
            ikmq = lookup(kmq, grid)
            if ikmq is not None:
                fq_direct[i] += trial[ikmq, 0] * Gh[0, j]

    trial_grid = trial[:, 0].reshape(mesh)
    Gh_grid = numpy.flip(Gh[0, :]).reshape(mesh)

    # Check by fft convolve
    # Compare to fq
    fq_conv = numpy.zeros(nqgrid, dtype=numpy.complex128)
    fq_conv += nqgrid * convolve(trial_grid, Gh_grid, mesh)
    fq_conv = numpy.flip(fq_conv)

    import scipy.signal

    fq_conv_sc = numpy.flip(scipy.signal.fftconvolve(trial_grid, Gh_grid).ravel())

    import matplotlib.pyplot as pl

    pl.plot(fq_conv, label="fft")
    pl.plot(fq_conv_sc, label="fft_scipy")
    pl.plot(fq_direct, label="direct")
    # pl.plot(fq, label='from_gf')
    pl.legend()
    pl.show()
    # print(Gtrace, Gtrace_direct, fq[1])


if __name__ == "__main__":
    # test_fft_kmq(2)
    test_fft_kpq(2)

    # import scipy
    # import scipy.signal
    # ra = numpy.random.random((3,3))
    # za = numpy.random.random((3,3))
    # rb = numpy.random.random((3,3))
    # zb = numpy.random.random((3,3))

    # a = ra + 1j * za
    # b = rb + 1j * zb

    # g = numpy.zeros((5*5), dtype=numpy.complex128) # 0 -1 1 -2 2

    # b1 = numpy.array([-1, 0, 1])
    # q1 = numpy.array([-2, -1, 0, 1, 2])
    # basis =  numpy.array(list(itertools.product(*[b1,b1])), dtype=numpy.int32)
    # qbasis = numpy.array(list(itertools.product(*[q1,q1])), dtype=numpy.int32)
    # print(qbasis.shape)

    # fq_conv_sc = scipy.signal.fftconvolve(a,b, mode='full').ravel()
    # print(fq_conv_sc.shape)

    # a = a.reshape(9)
    # b = b.reshape(9)

    # # \sum_j a[j-i] * b[j]
    # for i, q in enumerate(qbasis):
    #     for j, k in enumerate(basis):
    #         transfer = q-k
    #         idx = lookup (transfer, basis)
    #         if (idx != None):
    #             g[i] += a[j] * b[idx]

    # print(fq_conv_sc-g)
