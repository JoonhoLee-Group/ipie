import itertools
import os
import unittest

import numpy
import pyfftw
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
    kval = numpy.array(list(itertools.product(*[gx])), dtype=numpy.int32)
    spval = 0.5 * numpy.array([numpy.dot(g, g) for g in kval])

    return kval, spval


def lookup(g, basis):
    for i, k in enumerate(basis):
        if numpy.dot(g - k, g - k) == 0:
            return i
    return None


# Gives same results as scipy.signal.convolve
# def convolve(f, g, mesh):
#     f_ = f.reshape(*mesh)
#     g_ = g.reshape(*mesh)
#     shape = numpy.maximum(f_.shape, g_.shape)
#     min_shape = numpy.array(f_.shape) + numpy.array(g_.shape) - 1
#     fshape = [next_fast_len(d) for d in min_shape]
#     fslice = tuple([slice(sz) for sz in shape])
#     fq = numpy.fft.fftn(numpy.fft.ifftn(f_, s=fshape) *
#                         numpy.fft.ifftn(g_, s=fshape)).copy()
#     return fq.ravel()


def convolve(f, g, mesh):
    f_ = f.reshape(*mesh)
    g_ = g.reshape(*mesh)
    shape = numpy.maximum(f_.shape, g_.shape)
    min_shape = numpy.array(f_.shape) + numpy.array(g_.shape) - 1
    fshape = [next_fast_len(d) for d in min_shape]
    fslice = tuple([slice(sz) for sz in shape])
    fq = pyfftw.interfaces.numpy_fft.fftn(
        pyfftw.interfaces.numpy_fft.ifftn(f_, s=fshape)
        * pyfftw.interfaces.numpy_fft.ifftn(g_, s=fshape)
    ).copy()
    return fq.ravel()


def test_fft_kpq(nalpha):
    # Create regular grid.
    nmax = 2
    mesh = [2 * nmax + 1] * 1
    grid, eigs = generate_fft_grid(mesh)

    qmax = 2 * nmax
    qmesh = [2 * qmax + 1] * 1
    qgrid, qeigs = generate_fft_grid(qmesh)

    # Create wavefunction
    nbasis = len(grid)

    numpy.random.seed(7)

    psi = get_random_wavefunction((nalpha, nalpha), nbasis)
    I = numpy.eye(nbasis, dtype=numpy.complex128)
    # print(I.shape)
    I = get_random_wavefunction((nalpha, nalpha), nbasis)
    # print(I.shape)
    # exit()

    # Select lowest energy states for trial
    trial = I[:, :nalpha].conj()
    trial.imag[:] = 0
    G, Gh = gab_mod(trial, psi[:, :nalpha])
    nqgrid = numpy.prod(qmesh)

    # # Check by direct convolution f(q) = \sum_G Psi[G+Q] Gh[G].
    print(grid.shape)
    fq_direct = numpy.zeros(nqgrid, dtype=numpy.complex128)
    for iq, q in enumerate(qgrid):
        Gtrace_direct = 0
        # for a in range(nalpha):
        # compute \sum_G f_G(q)
        for i, k in enumerate(grid):
            kpq = k + q
            ikpq = lookup(kpq, grid)
            if ikpq is not None:
                Gtrace_direct += trial[ikpq, 0] * Gh[0, i]
        fq_direct[iq] = Gtrace_direct

    # Check by fft convolve
    # Compare to fq
    fq_conv = numpy.zeros(nqgrid, dtype=numpy.complex128)
    trial_pq = trial[:, 0].copy()
    Gh_pq = Gh[0, :].copy()

    fq_conv += nqgrid * convolve(Gh_pq, numpy.flip(trial_pq), mesh)
    fq_conv = numpy.flip(fq_conv)

    import scipy.signal

    fq_conv_sc = numpy.flip(
        scipy.signal.fftconvolve(Gh_pq, numpy.flip(trial_pq)).ravel()
    )

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
    mesh = [2 * nmax + 1] * 1
    grid, eigs = generate_fft_grid(mesh)

    qmax = 2 * nmax
    qmesh = [2 * qmax + 1] * 1
    qgrid, qeigs = generate_fft_grid(qmesh)

    # Create wavefunction
    nbasis = len(grid)

    numpy.random.seed(7)

    psi = get_random_wavefunction((nalpha, nalpha), nbasis)
    I = numpy.eye(nbasis, dtype=numpy.complex128)
    # print(I.shape)
    I = get_random_wavefunction((nalpha, nalpha), nbasis)
    # print(I.shape)
    # exit()

    # Select lowest energy states for trial
    trial = I[:, :nalpha].conj()
    trial.imag[:] = 0

    G, Gh = gab_mod(trial, psi[:, :nalpha])
    nqgrid = numpy.prod(qmesh)

    # # Check by direct convolution f(q) = \sum_G Psi[G+Q] Gh[G].
    print(grid.shape)
    fq_direct = numpy.zeros(nqgrid, dtype=numpy.complex128)
    for iq, q in enumerate(qgrid):
        Gtrace_direct = 0
        # for a in range(nalpha):
        # compute \sum_G f_G(q)
        for i, k in enumerate(grid):
            # kmq = q-k
            kmq = k - q
            ikmq = lookup(kmq, grid)
            # idx = numpy.argwhere(basis == q-k)
            # print(idx, ikmq)
            if ikmq is not None:
                Gtrace_direct += trial[i, 0] * Gh[0, ikmq]
        fq_direct[iq] = Gtrace_direct
    print("trial[igmq,0] = {}".format(trial[:, 0]))
    print("Gh[0,i] = {}".format(Gh[0, :]))
    # Check by fft convolve
    # Compare to fq
    fq_conv = numpy.zeros(nqgrid, dtype=numpy.complex128)
    trial_pq = trial[:, 0].copy()
    Gh_pq = Gh[0, :].copy()

    # \sum_G f(G-Q) g(G)
    # -G-Q -> G'
    # \sum_G g(-G-Q) f(-G) = \sum_G g(G) f(G+Q)
    # trial_pq = numpy.conj(trial_pq)
    fq_conv += nqgrid * convolve(Gh_pq, numpy.flip(trial_pq), mesh)
    fq_conv = numpy.flip(fq_conv)

    import scipy.signal

    fq_conv_sc = numpy.flip(
        scipy.signal.fftconvolve(Gh_pq, numpy.flip(trial_pq)).ravel()
    )

    # for i in range(qmesh[0]):
    #     target = fq_conv[i]
    #     print(numpy.abs(target))
    #     if (numpy.abs(target) > 1e-8):
    #         # print(target)
    #         idx = numpy.argwhere(numpy.abs(fq_direct - target) < 1e-8)
    #         # print(i, idx)

    import matplotlib.pyplot as pl

    pl.plot(fq_conv, label="fft")
    pl.plot(fq_conv_sc, label="fft_scipy")
    pl.plot(fq_direct, label="direct")
    # pl.plot(fq, label='from_gf')
    pl.legend()
    pl.show()
    # print(Gtrace, Gtrace_direct, fq[1])


if __name__ == "__main__":
    import scipy
    import scipy.signal

    # test_fft_kmq(2)
    test_fft_kpq(2)
    # ra = numpy.random.rand(3)
    # za = numpy.random.rand(3)
    # rb = numpy.random.rand(3)
    # zb = numpy.random.rand(3)

    # # a = ra + 1j * za
    # # b = rb + 1j * zb
    # a = numpy.array([0.90912837-0.76864751*1j, 0.66901324-0.45284293*1j, 0.37238469-0.45909298*1j])
    # b = numpy.array([ 1.32048196-0.35323835*1j, -1.17366914-0.54122172*1j,  0.33498304+0.12336102*1j])
    # # a = numpy.array([1, 2, 3]) # 0 -1 1
    # # b = numpy.array([0, 1, 0.5]) # 0 -1 1
    # # # g = numpy.convolve(a, b)
    # g = numpy.zeros(5, dtype=numpy.complex128) # 0 -1 1 -2 2

    # # basis =  numpy.array([0, -1, 1])
    # # qbasis = numpy.array([0, -1, 1, -2, 2])
    # # gx = numpy.fft.fftfreq(3, 1.0/3.)
    # # print(gx)
    # basis =  numpy.array([-1, 0, 1])
    # qbasis = numpy.array([-2, -1, 0, 1, 2])

    # fq_conv_sc = scipy.signal.fftconvolve(a,
    #         b).ravel()
    # print(fq_conv_sc)

    # # \sum_j a[j-i] * b[j]
    # for i, q in enumerate(qbasis):
    #     for j, k in enumerate(basis):
    #         if (k-q <= 1 and k-q >= -1):
    #             idx = numpy.argwhere(basis == q-k)
    #             g[i] += a[j] * b[idx]
    # print(g)
