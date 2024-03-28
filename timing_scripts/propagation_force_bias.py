import os

# numpy.show_config()
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def force_bias_numba(rchol, walkers_batch_Ghalf):
    walkers_batch_Ghalfa_real_T = walkers_batch_Ghalf[0].real.T.copy()
    walkers_batch_Ghalfa_imag_T = walkers_batch_Ghalf[0].imag.T.copy()
    walkers_batch_Ghalfb_real_T = walkers_batch_Ghalf[1].real.T.copy()
    walkers_batch_Ghalfb_imag_T = walkers_batch_Ghalf[1].imag.T.copy()
    vfb2_real = rchol.dot(walkers_batch_Ghalfa_real_T) + rchol.dot(walkers_batch_Ghalfb_real_T)
    vfb2_imag = rchol.dot(walkers_batch_Ghalfa_imag_T) + rchol.dot(walkers_batch_Ghalfb_imag_T)
    vfb2 = np.zeros_like(vfb2_real, dtype=np.complex128)
    vfb2 = vfb2_real + 1.0j * vfb2_imag
    vfb2 = vfb2.T.copy()
    return vfb2


divide = 2

nao = 1000 // divide
nocc = 200 // divide
naux = 4000 // divide
nwalkers = 20

rchol = np.random.rand(naux, nocc * nao)
walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao, nocc) + 1.0j * np.random.rand(
    2, nwalkers, nao, nocc
)

# algorithm 1
t0 = time.time()
vfb = []
for iw in range(nwalkers):
    vfb += [
        np.dot(rchol, walkers_batch_Ghalf[0][iw].ravel())
        + np.dot(rchol, walkers_batch_Ghalf[1][iw].ravel())
    ]
vfb = np.array(vfb)
t1 = time.time()
print("forming vfb naive = {}".format(t1 - t0))

walkers_batch_Ghalf = walkers_batch_Ghalf.reshape(2, nwalkers, nao * nocc)
t0 = time.time()
vfb2 = rchol.dot(walkers_batch_Ghalf[0].T) + rchol.dot(walkers_batch_Ghalf[1].T)
vfb2 = vfb2.T.copy()
t1 = time.time()
print("forming vfb combined = {}".format(t1 - t0))

assert np.allclose(vfb2, vfb)

walkers_batch_Ghalf = walkers_batch_Ghalf.reshape(2, nwalkers, nao * nocc)
t0 = time.time()
vfb2_real = rchol.dot(walkers_batch_Ghalf[0].real.T) + rchol.dot(walkers_batch_Ghalf[1].real.T)
vfb2_imag = rchol.dot(walkers_batch_Ghalf[0].imag.T) + rchol.dot(walkers_batch_Ghalf[1].imag.T)
vfb2 = np.zeros_like(vfb2_real, dtype=np.complex128)
vfb2.real = vfb2_real
vfb2.imag = vfb2_imag
vfb2 = vfb2.T.copy()
t1 = time.time()
print("forming vfb combined split complex = {}".format(t1 - t0))

assert np.allclose(vfb2, vfb)


walkers_batch_Ghalf = walkers_batch_Ghalf.reshape(2, nwalkers, nao * nocc)

t0 = time.time()
walkers_batch_Ghalfa_real_T = walkers_batch_Ghalf[0].real.T.copy()
walkers_batch_Ghalfa_imag_T = walkers_batch_Ghalf[0].imag.T.copy()
walkers_batch_Ghalfb_real_T = walkers_batch_Ghalf[1].real.T.copy()
walkers_batch_Ghalfb_imag_T = walkers_batch_Ghalf[1].imag.T.copy()
vfb2_real = rchol.dot(walkers_batch_Ghalfa_real_T) + rchol.dot(walkers_batch_Ghalfb_real_T)
vfb2_imag = rchol.dot(walkers_batch_Ghalfa_imag_T) + rchol.dot(walkers_batch_Ghalfb_imag_T)
vfb2 = np.zeros_like(vfb2_real, dtype=np.complex128)
vfb2.real = vfb2_real
vfb2.imag = vfb2_imag
vfb2 = vfb2.T.copy()
t1 = time.time()
print("forming vfb combined split complex + contiguous = {}".format(t1 - t0))

assert np.allclose(vfb2, vfb)


# t0 = time.time()
vfb3 = force_bias_numba(rchol, walkers_batch_Ghalf)
# t1 = time.time()
# print("forming vfb numba = {}".format(t1 - t0))
# assert np.allclose(vfb3, vfb)

t0 = time.time()
vfb3 = force_bias_numba(rchol, walkers_batch_Ghalf)
t1 = time.time()
print("forming vfb numba = {}".format(t1 - t0))
assert np.allclose(vfb3, vfb)
