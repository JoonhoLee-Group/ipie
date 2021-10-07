import numpy as np
import time
# import cProfile, pstats, io
# from pstats import SortKey

divide = 1

nao = 1000 // divide
nocc = 200  // divide
naux = 4000  // divide
nwalkers = 20

chol = np.random.rand(naux, nao, nao)
chol2 = np.random.rand(nao, nao, naux)
chol3 = np.random.rand(naux,nao*nao)
x = np.random.rand(nwalkers, naux)

"""1"""
t0 = time.time()
VHS = []
for i in range(nwalkers):
    VHS += [chol2.dot(x[i])]
t1 = time.time()
print("forming VHS naive = {}".format(t1 - t0))

"""2"""
t0 = time.time()
VHS = np.einsum("wX,Xmn->wmn", x, chol, optimize=True)
t1 = time.time()
print("forming VHS combined = {}".format(t1 - t0))
#
"""3"""
t0 = time.time()
VHS = x.dot(chol3)
VHS = VHS.reshape((nwalkers, nao, nao))
t1 = time.time()
print("forming VHS combined 3 = {}".format(t1 - t0))
#
