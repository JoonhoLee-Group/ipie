import math
import time

import cupy
import numpy
from numba import cuda, vectorize

from ipie.utils.pack import pack_cholesky, pack_cholesky_fast, unpack_VHS_batch


@cuda.jit('void(int32[:],int32[:],complex128[:,:],complex128[:,:,:])',device=False)
def unpack_VHS_batch_gpu(idx_i,idx_j,VHS_packed,VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf *(nbsf+1)/2)
    pos = cuda.grid(1)
    pos1 = pos // nut
    pos2 = pos - pos1 * nut
    if (pos1 < nwalkers and pos2 < nut):
        VHS[pos1, idx_i[pos2],idx_j[pos2]] = VHS_packed[pos1,pos2]
        VHS[pos1, idx_j[pos2],idx_i[pos2]] = VHS_packed[pos1,pos2]

nbsf = 800
nwalkers = 100
nchol = 4*nbsf
nupper = int(nbsf*(nbsf+1)/2)

xauxf = numpy.random.random((nchol,nwalkers)) + 1.j * numpy.random.random((nchol,nwalkers))

start_time = time.time()
Lchol = numpy.random.randn(nbsf**2*nchol).reshape(nbsf,nbsf, nchol)
Lchol_packed = numpy.zeros((nupper,nchol))
end_time = time.time()
#print("allocation: {}".format(end_time-start_time))

start_time = time.time()
Lchol = Lchol.transpose(0,1,2) + Lchol.transpose(1,0,2)
end_time = time.time()
#print("symmetrization: {}".format(end_time-start_time))

idx = numpy.triu_indices(nbsf)

start_time = time.time()

pack_cholesky(idx[0],idx[1],Lchol_packed, Lchol)

end_time = time.time()
#print("packing: {}".format(end_time-start_time))

Lchol = Lchol.reshape(nbsf**2,nchol)
start_time = time.time()
VHS = Lchol.dot(xauxf.real) + 1.j * Lchol.dot(xauxf.imag)
VHS = VHS.T.copy()
VHS = VHS.reshape(nwalkers,nbsf,nbsf)
end_time = time.time()
print("original: {}".format(end_time-start_time))

start_time = time.time()
VHS_packed = Lchol_packed.dot(xauxf.real) + 1.j * Lchol_packed.dot(xauxf.imag)
VHS_packed = VHS_packed.T.copy()
VHS_packed_CPU = VHS_packed.copy()
end_time = time.time()
tpacked_gemm = end_time - start_time
print("packed gemm (CPU): {}".format(tpacked_gemm))

start_time = time.time()
VHS_unpacked = numpy.zeros((nwalkers,nbsf,nbsf),dtype=numpy.complex128)
unpack_VHS_batch(idx[0],idx[1],VHS_packed,VHS_unpacked)
end_time = time.time()
tunpacking = end_time-start_time
print("unpacking (CPU): {}".format(tunpacking))
print("packed total: {}".format(tpacked_gemm + tunpacking))

A = cupy.zeros((2,2),dtype=cupy.complex128)
B = A.dot(A)
B = A.T.dot(A)
B = A.T.copy().dot(A)

idx_i = cupy.array(idx[0])
idx_j = cupy.array(idx[1])
Lchol_packed = cupy.array(Lchol_packed)
xauxf = cupy.array(xauxf)

VHS_packed = Lchol_packed.dot(xauxf.real) + 1.j * Lchol_packed.dot(xauxf.imag)
VHS_packed = VHS_packed.T.copy()
cupy.cuda.Stream.null.synchronize()
start_time = time.time()
VHS_packed = Lchol_packed.dot(xauxf.real) + 1.j * Lchol_packed.dot(xauxf.imag)
VHS_packed = VHS_packed.T.copy()
end_time = time.time()
tpacked_gemm = end_time - start_time
print("packed gemm (GPU, DP): {}".format(tpacked_gemm))

#Lchol_packed = Lchol_packed.astype(numpy.float32)
#xauxf = xauxf.astype(numpy.complex64)
#VHS_packed = Lchol_packed.dot(xauxf.real) + 1.j * Lchol_packed.dot(xauxf.imag)
#VHS_packed = VHS_packed.T.copy()
#start_time = time.time()
#VHS_packed = Lchol_packed.dot(xauxf.real) + 1.j * Lchol_packed.dot(xauxf.imag)
#VHS_packed = VHS_packed.T.copy()
#end_time = time.time()
#tpacked_gemm = end_time - start_time
#print("packed gemm (GPU, SP): {}".format(tpacked_gemm))

VHS_unpacked = cupy.zeros((nwalkers,nbsf,nbsf),dtype=numpy.complex128)

threadsperblock = 512
nut = round(nbsf *(nbsf+1)/2)
blockspergrid = math.ceil(VHS.shape[0]*nut / threadsperblock)
unpack_VHS_batch_gpu[blockspergrid, threadsperblock](idx_i,idx_j,VHS_packed,VHS_unpacked)
cupy.cuda.Stream.null.synchronize()
start_time = time.time()
unpack_VHS_batch_gpu[blockspergrid, threadsperblock](idx_i,idx_j,VHS_packed,VHS_unpacked)
end_time = time.time()
print("unpacking (GPU): {}".format(end_time-start_time))

diff = VHS - cupy.asnumpy(VHS_unpacked)
print(numpy.max(numpy.abs(diff)))

diff = VHS_packed_CPU - cupy.asnumpy(VHS_packed)
print(numpy.max(numpy.abs(diff)))
