import h5py
import numpy
from pie.utils.io import write_qmcpack_dense, write_qmcpack_sparse, write_qmcpack_wfn
from pie.utils.linalg import modified_cholesky

f = h5py.File("2FeIII_nat.h5", "r")
h1e = f["h1e"][()]
h2e = f["h2e"][()]
ecore = f["ecore"][()]
coeff0 = numpy.array(f.get("ci"))
occa_ref = numpy.array(f.get("occa"))
occb_ref = numpy.array(f.get("occb"))
f.close()
print(len(occa_ref))

ndets = [464722]
for ndet in ndets:
    coeff = numpy.array(coeff0[:ndet], dtype=numpy.complex128)
    occa = occa_ref[:ndet]
    occb = occb_ref[:ndet]
    for i in range(ndet):
        doubles = list(set(occa[i]) & set(occb[i]))
        occa0 = numpy.array(occa[i])
        occb0 = numpy.array(occb[i])

        count = 0
        for ocb in occb0:
            passing_alpha = numpy.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phase = (-1) ** count
        coeff[i] *= phase

    print("ordered coeff = {}".format(coeff))

    na = len(occa[0])
    nb = len(occb[0])
    nmo = h1e.shape[-1]

    h2e = h2e.reshape((nmo * nmo, nmo * nmo))

    e, v = numpy.linalg.eigh(h2e)

    idx = e.argsort()[::-1]
    e = e[idx]
    v = v[:, idx]
    idx2 = e > 1e-7
    e = e[idx2]
    v = v[:, idx2].dot(numpy.diag(numpy.sqrt(e)))
    nchol = len(e)

    h1e = numpy.array(h1e, dtype=numpy.complex128)
    chol = numpy.array(v, dtype=numpy.complex128)

    write_qmcpack_sparse(
        h1e,
        chol.copy(),
        (na, nb),
        nmo,
        ecore,
        filename="afqmc_sparse.h5",
        real_chol=False,
        verbose=True,
        ortho=None,
    )
    coeff = numpy.array(coeff, dtype=numpy.complex128)
    # Sort in ascending order.
    ixs = numpy.argsort(numpy.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occa = numpy.array(occa)[ixs]
    occb = numpy.array(occb)[ixs]

    uhf = True
    write_qmcpack_wfn("wfn_phases.h5", (coeff, occa, occb), uhf, (na, nb), nmo)
