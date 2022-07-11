import numpy


def fft_wavefunction(psi, nx, ny, ns, sin):
    return numpy.fft.fft2(psi.reshape(nx, ny, ns), axes=(0, 1)).reshape(sin)


def ifft_wavefunction(psi, nx, ny, ns, sin):
    return numpy.fft.ifft2(psi.reshape(nx, ny, ns), axes=(0, 1)).reshape(sin)
