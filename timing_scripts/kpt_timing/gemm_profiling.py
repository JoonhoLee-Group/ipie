import numpy as np
from numba import jit
import time


def apply_exponential(phi, VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation
    Parameters
    ----------
    system :
        system class
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential
    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    Temp = np.zeros(phi.shape, dtype=phi.dtype)

    np.copyto(Temp, phi)
    for n in range(1, exp_nmax + 1):
        Temp = VHS.dot(Temp) / n
        phi += Temp

    return phi


def gemm(nwalkers, phia, phib, VHS, exp_nmax):
    for iw in range(nwalkers):
        phia[iw] = apply_exponential(phia[iw], VHS[iw], exp_nmax)
        phib[iw] = apply_exponential(phib[iw], VHS[iw], exp_nmax)


nwalkers = 10
nbsf = 26
nk = 27
nup, ndown = 4, 4
exp_nmax = 6

elapsed_time_lis = []

for i in range(100):
    phia = np.random.rand(nwalkers, nk * nbsf, nk * nup) + 1j * np.random.rand(
        nwalkers, nk * nbsf, nk * nup
    )
    phib = np.random.rand(nwalkers, nk * nbsf, nk * ndown) + 1j * np.random.rand(
        nwalkers, nk * nbsf, nk * ndown
    )
    VHS = np.random.rand(nwalkers, nk * nbsf, nk * nbsf) + 1j * np.random.rand(
        nwalkers, nk * nbsf, nk * nbsf
    )
    stttime = time.time()
    gemm(nwalkers, phia, phib, VHS, exp_nmax)
    # print("Elapsed time: ", time.time()-stttime)
    elapsed_time = time.time() - stttime
    elapsed_time_lis.append(elapsed_time)

elapsed_time_lis = np.array(elapsed_time_lis)
print("Elapsed time: ", elapsed_time_lis)
print("Mean elapsed time: ", np.mean(elapsed_time_lis))
print("Standard deviation: ", np.std(elapsed_time_lis))
