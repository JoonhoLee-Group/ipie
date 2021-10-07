import numpy
import time
# import cProfile, pstats, io
# from pstats import SortKey

divide = 5

nao = 1000 // divide
naux = 4000  // divide
nwalkers = 20

chol = numpy.random.rand(naux, nao, nao)
chol = chol + chol.transpose(0,2,1)
chol3 = chol.reshape(naux,nao*nao).copy()
chol2 = chol.T.reshape(nao,nao,naux).copy() #numpy.random.rand(nao, nao, naux)
x = numpy.random.rand(nwalkers, naux)

"""1"""
t0 = time.time()
VHS1 = []
for i in range(nwalkers):
    VHS1 += [chol2.dot(x[i])]
t1 = time.time()
VHS1 = numpy.array(VHS1)
print("forming VHS1 naive = {}".format(t1 - t0))

print(VHS1.shape)
# print(VHS1)

"""2"""
t0 = time.time()
VHS2 = numpy.einsum("wX,Xu->wu", x, chol3, optimize=True)
VHS2 = VHS2.reshape((nwalkers, nao, nao))
t1 = time.time()
print("forming VHS2 combined = {}".format(t1 - t0))
print(VHS2.shape)
# print(VHS2)
#
"""3"""
t0 = time.time()
VHS3 = x.dot(chol3)
VHS3 = VHS3.reshape((nwalkers, nao, nao))
t1 = time.time()
print("forming VHS3 combined 3 = {}".format(t1 - t0))
#
assert numpy.allclose(VHS1, VHS2)
assert numpy.allclose(VHS2, VHS3)
