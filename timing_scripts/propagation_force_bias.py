import numpy as np
import time

divide = 5

nao = 1000 // divide
nocc = 200  // divide
naux = 4000  // divide
nwalkers = 20

rchol = np.random.rand(naux,nocc*nao)
walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao, nocc)

# algorithm 1
t0 = time.time()
vfb = []
for iw in range(nwalkers):
    vfb += [np.dot(rchol, walkers_batch_Ghalf[0][iw].ravel()) + np.dot(rchol, walkers_batch_Ghalf[1][iw].ravel())]
vfb = np.array(vfb)
t1 = time.time()
print("forming vfb naive = {}".format(t1 - t0))

t0 = time.time()
walkers_batch_Ghalf = walkers_batch_Ghalf.reshape(2, nwalkers, nao*nocc)
vfb2 = rchol.dot(walkers_batch_Ghalf[0].T) + rchol.dot(walkers_batch_Ghalf[1].T)
vfb2 = vfb2.T.copy()
t1 = time.time()
print("forming vfb combined = {}".format(t1 - t0))

assert np.allclose(vfb2, vfb)
