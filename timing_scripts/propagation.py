import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey

divide = 3

nao = 1000 // divide
nocc = 200  // divide
naux = 4000  // divide
nwalkers = 20

chol = np.random.rand(naux,nao*nao)
x = np.random.rand(nwalkers, naux)

t0 = time.time()
VHS = x.dot(chol)
VHS = VHS.reshape(nwalkers, nao, nao)
t1 = time.time()
print("forming VHS combined = {}".format(t1 - t0))

walkers_batch_phi0 = np.random.rand(nwalkers, 2, nao, nocc)
# version 1
walkers_batch_phi = walkers_batch_phi0.copy()
t0 = time.time()
for i in range (6):
   for iw in range(nwalkers):
       walkers_batch_phi[iw][0] = VHS[iw].dot(walkers_batch_phi[iw][0])
       walkers_batch_phi[iw][1] = VHS[iw].dot(walkers_batch_phi[iw][1])
t1 = time.time()
print("naive propagation = {}".format((t1 - t0)))

# version 2
walkers_batch_phi2 = walkers_batch_phi0.copy()
walkers_batch_phi2 = np.einsum("swmi->wsmi", walkers_batch_phi2)
t0 = time.time()
for i in range (6):
    walkers_batch_phi2[0] = np.einsum("wmn,wni->wmi", VHS, walkers_batch_phi2[0], optimize=True)
    walkers_batch_phi2[1] = np.einsum("wmn,wni->wmi", VHS, walkers_batch_phi2[1], optimize=True)
t1 = time.time()
walkers_batch_phi2 = np.einsum("swmi->wsmi", walkers_batch_phi2)
print(walkers_batch_phi2.shape, walkers_batch_phi.shape)
assert np.allclose(walkers_batch_phi[0],walkers_batch_phi2[0])
assert np.allclose(walkers_batch_phi[1],walkers_batch_phi2[1])
print("propagation 2 = {}".format((t1 - t0)))


