import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey

divide = 5

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

walkers_batch_phi0 = np.random.rand(nwalkers, nao, nocc)

# version 1
walkers_batch_phi = walkers_batch_phi0.copy()
t0 = time.time()
for i in range (6):
   for iw in range(nwalkers):
       walkers_batch_phi[iw] = VHS[iw].dot(walkers_batch_phi[iw])
t1 = time.time()
print("naive propagation = {}".format((t1 - t0)))

# version 2
walkers_batch_phi2 = walkers_batch_phi0.copy()
t0 = time.time()
for i in range (6):
    walkers_batch_phi2 = np.einsum("wmn,wni->wmi", VHS, walkers_batch_phi2, optimize=True)
t1 = time.time()
assert np.allclose(walkers_batch_phi[0],walkers_batch_phi2[0])
print("propagation 2 = {}".format((t1 - t0)))

# # version 3
# walkers_batch_phi3 = np.hstack((w for w in walkers_batch_phi0)).reshape(nao, nwalkers, nocc)
# print (walkers_batch_phi3.shape, walkers_batch_phi0.shape)
# VHS.dot(walkers_batch_phi3)
# # # version 1
# # walkers_batch_phi = walkers_batch_phi0.copy()
# # t0 = time.time()
# # for i in range (6):
# #    for iw in range(nwalkers):
# #        walkers_batch_phi[iw][0] = VHS[iw].dot(walkers_batch_phi[iw][0])
# #        walkers_batch_phi[iw][1] = VHS[iw].dot(walkers_batch_phi[iw][1])
# # t1 = time.time()
# # print("propagation 3 = {}".format((t1 - t0)))





