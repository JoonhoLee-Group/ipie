import itertools
import math

import numpy
import scipy
import scipy.linalg
import scipy.sparse.linalg


def simple_fci_bose_fermi(
    system, ham, nboson_max=1, gen_dets=False, occs=None, hamil=False, verbose=False
):
    """Very dumb FCI routine."""
    orbs = numpy.arange(system.nbasis)

    # bosons
    blkboson = [1]  # blk size for each boson sector
    perms = [[0 for i in range(system.nbasis)]]
    for nboson in range(1, nboson_max + 1):
        perm = list(
            unlabeled_balls_in_labeled_boxes(
                nboson, [nboson for i in range(system.nbasis)]
            )
        )
        perms += perm
        blkboson += [len(perm)]
    # print("blkboson = {}".format(blkboson))
    nperms = len(perms)
    for i, perm in enumerate(perms):
        perms[i] = numpy.array(perm)

    if occs is None:
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        oa, ob = zip(*itertools.product(oa, ob))
    else:
        oa, ob = occs

    # convert to spin orbitals
    dets = [[j for j in a] + [i + system.nbasis for i in c] for (a, c) in zip(oa, ob)]
    dets = [numpy.sort(d) for d in dets]
    ndets = len(dets)

    print("# ndets, nperms, ntot = {}, {}, {}".format(ndets, nperms, ndets * nperms))

    ntot = ndets * nperms

    # Htot = numpy.zeros((ndets*nperms, ndets*nperms))
    Htot = scipy.sparse.csr_matrix((ndets * nperms, ndets * nperms))

    Iel = scipy.sparse.eye(ndets)
    Ib = scipy.sparse.eye(nperms)

    hel = scipy.sparse.csr_matrix((ndets, ndets))
    nel = system.nup + system.ndown
    for i in range(ndets):
        for j in range(i, ndets):
            hel[i, j] = get_hmatel(system, nel, dets[i], dets[j])[0]
            hel[j, i] = hel[i, j]

    print("# finshed forming hel")

    hb = scipy.sparse.csr_matrix((nperms, nperms))

    for i in range(nperms):
        p = numpy.asarray(perms[i])
        nocc = numpy.sum(p)
        hb[i, i] = system.w0 * nocc

    print("# finshed forming hb")

    Heb = scipy.sparse.csr_matrix(Htot.shape)
    for isite in range(system.nbasis):
        rhoi = scipy.sparse.csr_matrix((ndets, ndets))
        for i, di in enumerate(dets):
            for d in di:
                ii, spin_ii = map_orb(d, system.nbasis)
                if ii == isite:
                    rhoi[i, i] += 1.0

        bi = scipy.sparse.csr_matrix((nperms, nperms))
        for i, iperm in enumerate(perms):
            ni = numpy.sum(iperm)
            offset_i = numpy.sum(blkboson[: ni + 1])  # block size sum
            if ni == nboson_max:
                continue

            for j, jperm in enumerate(perms[offset_i : offset_i + blkboson[ni + 1]]):
                diff = numpy.array(iperm) - numpy.array(jperm)
                ndiff = numpy.sum(numpy.abs(diff))
                if ndiff == 1 and diff[isite] == -1:
                    factor = math.sqrt(numpy.array(iperm)[isite] + 1)
                    bi[i, j + offset_i] = 1.0 * factor

        xi = bi + bi.T

        srhoi = scipy.sparse.csr_matrix(rhoi)
        sxi = scipy.sparse.csr_matrix(xi)

        Heb += system.g * scipy.sparse.kron(sxi, srhoi)

    print("# finshed forming Heb")

    He = scipy.sparse.kron(Ib, hel)
    Hb = scipy.sparse.kron(hb, Iel)

    Htot = He + Hb + Heb

    print("# finshed forming Htot")
    print(
        "# He nnz = {} out of total {}".format(He.nnz, ndets * nperms * ndets * nperms)
    )
    print(
        "# Hb nnz = {} out of total {}".format(Hb.nnz, ndets * nperms * ndets * nperms)
    )
    print(
        "# Heb nnz = {} out of total {}".format(
            Heb.nnz, ndets * nperms * ndets * nperms
        )
    )
    print(
        "# Htot nnz = {} out of total {}".format(
            Htot.nnz, ndets * nperms * ndets * nperms
        )
    )

    eigval, eigvec = scipy.sparse.linalg.eigsh(Htot, k=3, which="SA")

    Eel = eigvec[:, 0].T.conj().dot(He.dot(eigvec[:, 0]))
    Eb = eigvec[:, 0].T.conj().dot(Hb.dot(eigvec[:, 0]))
    Eeb = eigvec[:, 0].T.conj().dot(Heb.dot(eigvec[:, 0]))

    if verbose:
        for isite in range(system.nbasis):
            rhoi = scipy.sparse.csr_matrix((ndets, ndets))
            for i, di in enumerate(dets):
                for d in di:
                    ii, spin_ii = map_orb(d, system.nbasis)
                    if ii == isite:
                        rhoi[i, i] += 1.0
            rho = scipy.sparse.kron(Ib, rhoi)
            nocc1 = eigvec[:, 0].T.conj().dot(rho.dot(eigvec[:, 0]))
            print("i, nocc = {}, {}".format(isite, nocc1))

        for isite in range(system.nbasis):
            bi = scipy.sparse.csr_matrix((nperms, nperms))
            for i, iperm in enumerate(perms):
                ni = numpy.sum(iperm)
                offset_i = numpy.sum(blkboson[: ni + 1])  # block size sum
                if ni == nboson_max:
                    continue

                for j, jperm in enumerate(
                    perms[offset_i : offset_i + blkboson[ni + 1]]
                ):
                    diff = numpy.array(iperm) - numpy.array(jperm)
                    ndiff = numpy.sum(numpy.abs(diff))
                    if ndiff == 1 and diff[isite] == -1:
                        factor = math.sqrt(numpy.array(iperm)[isite] + 1)
                        bi[i, j + offset_i] = 1.0 * factor

            nib = bi.T.dot(bi)
            ni = scipy.sparse.kron(nib, Iel)

            xib = (bi + bi.T) / numpy.sqrt(2.0 * system.m * system.w0)
            xi = scipy.sparse.kron(xib, Iel)

            X = eigvec[:, 0].T.conj().dot(xi.dot(eigvec[:, 0]))
            print("i, X = {}, {}".format(isite, X))

        print(
            "# Eel, Eb, Eeb, Etot = {}, {}, {}, {}".format(Eel, Eb, Eeb, Eel + Eb + Eeb)
        )

    if gen_dets:
        return (eigval, eigvec), (dets, numpy.array(oa), numpy.array(ob))
    elif hamil:
        return (eigval, eigvec), Htot
    else:
        return (eigval, eigvec)


def simple_fci(system, ham, gen_dets=False, occs=None, hamil=False):
    """Very dumb FCI routine."""
    orbs = numpy.arange(ham.nbasis)
    if occs is None:
        oa = [c for c in itertools.combinations(orbs, system.nup)]
        ob = [c for c in itertools.combinations(orbs, system.ndown)]
        oa, ob = zip(*itertools.product(oa, ob))
    else:
        oa, ob = occs
    # convert to spin orbitals
    dets = [[j for j in a] + [i + ham.nbasis for i in c] for (a, c) in zip(oa, ob)]
    dets = [numpy.sort(d) for d in dets]
    ndets = len(dets)
    H = numpy.zeros((ndets, ndets))
    nel = system.nup + system.ndown
    for i in range(ndets):
        for j in range(i, ndets):
            H[i, j] = get_hmatel(ham, nel, dets[i], dets[j])[0]
    if gen_dets:
        return scipy.linalg.eigh(H, lower=False), (
            dets,
            numpy.array(oa),
            numpy.array(ob),
        )
    elif hamil:
        return scipy.linalg.eigh(H, lower=False), H
    else:
        return scipy.linalg.eigh(H, lower=False)


def get_hmatel(ham, nel, di, dj):
    from_orb = list(set(dj) - set(di))
    to_orb = list(set(di) - set(dj))
    from_orb.sort()
    to_orb.sort()
    nex = len(from_orb)
    perm = get_perm(from_orb, to_orb, di, dj)
    if nex == 0:
        hmatel, e1b, e2b = slater_condon0(ham, di)
    elif nex == 1:
        i, si = map_orb(from_orb[0], ham.nbasis)
        a, sa = map_orb(to_orb[0], ham.nbasis)
        hmatel, e1b, e2b = slater_condon1(ham, nel, (i, si), (a, sa), di, perm)
    elif nex == 2:
        # < ij | ab > or < ij | ba >
        i, si = map_orb(from_orb[0], ham.nbasis)
        j, sj = map_orb(from_orb[1], ham.nbasis)
        a, sa = map_orb(to_orb[0], ham.nbasis)
        b, sb = map_orb(to_orb[1], ham.nbasis)
        hmatel = slater_condon2(ham, (i, si), (j, sj), (a, sa), (b, sb), perm)
        e1b = 0
        e2b = hmatel
    else:
        hmatel = 0.0
        e1b = 0.0
        e2b = 0.0
    return numpy.array([hmatel, e1b, e2b])


def get_perm(from_orb, to_orb, di, dj):
    """Determine sign of permutation needed to align two determinants.

    Stolen from HANDE.
    """
    nmove = 0
    perm = 0
    for o in from_orb:
        io = numpy.where(dj == o)[0]
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in to_orb:
        io = numpy.where(di == o)[0]
        perm += io - nmove
        nmove += 1
    return perm % 2 == 1


def slater_condon0(system, occs):
    e1b = 0.0
    e2b = 0.0
    e1b = system.ecore
    for i in range(len(occs)):
        ii, spin_ii = map_orb(occs[i], system.nbasis)
        # Todo: Update if H1 is ever spin dependent.
        e1b += system.H1[0, ii, ii]
        for j in range(i + 1, len(occs)):
            jj, spin_jj = map_orb(occs[j], system.nbasis)
            e2b += system.hijkl(ii, jj, ii, jj)
            if spin_ii == spin_jj:
                e2b -= system.hijkl(ii, jj, jj, ii)
    hmatel = e1b + e2b
    return hmatel, e1b, e2b


def slater_condon1(ham, nel, i, a, occs, perm):
    ii, si = i
    aa, sa = a
    e1b = ham.H1[0, ii, aa]
    e2b = 0
    for j in range(nel):
        # \sum_j <ij|aj> - <ij|ja>
        oj = occs[j]
        oj, soj = map_orb(oj, ham.nbasis)
        if 2 * oj + soj != 2 * ii + si:
            e2b += ham.hijkl(ii, oj, aa, oj)
            if soj == si:
                e2b -= ham.hijkl(ii, oj, oj, aa)
    hmatel = e1b + e2b
    if perm:
        return -hmatel, -e1b, -e2b
    else:
        return hmatel, e1b, e2b


def slater_condon2(system, i, j, a, b, perm):
    ii, si = i
    jj, sj = j
    aa, sa = a
    bb, sb = b
    hmatel = 0.0
    if si == sa:
        hmatel = system.hijkl(ii, jj, aa, bb)
    if si == sb:
        hmatel -= system.hijkl(ii, jj, bb, aa)
    if perm:
        return -hmatel
    else:
        return hmatel


def map_orb(orb, nbasis):
    """Map spin orbital to spatial index."""
    if orb // nbasis == 0:
        s = 0
        ix = orb
    else:
        s = 1
        ix = orb - nbasis
    return ix, s


def get_one_body_matel(ints, di, dj):
    from_orb = list(set(dj) - set(di))
    to_orb = list(set(di) - set(dj))
    nex = len(from_orb)
    perm = get_perm(from_orb, to_orb, di, dj)
    matel = 0.0
    if nex == 0:
        for i in range(len(di)):
            ii, spin_ii = map_orb(di[i], ints.shape[-1])
            matel += ints[ii, ii]
    elif nex == 1:
        i, si = map_orb(from_orb[0], ints.shape[-1])
        a, sa = map_orb(to_orb[0], ints.shape[-1])
        assert si == sa
        matel = ints[i, a]
    else:
        matel = 0.0
    if perm:
        return -matel
    else:
        return matel


def unlabeled_balls_in_labeled_boxes(balls, box_sizes):
    """
    OVERVIEW
    This function returns a generator that produces all distinct distributions of
    indistinguishable balls among labeled boxes with specified box sizes
    (capacities).  This is a generalization of the most common formulation of the
    problem, where each box is sufficiently large to accommodate all of the
    balls, and is an important example of a class of combinatorics problems
    called 'weak composition' problems.
    CONSTRUCTOR INPUTS
    n: the number of balls
    box_sizes: This argument is a list of length 1 or greater.  The length of
    the list corresponds to the number of boxes.  `box_sizes[i]` is a positive
    integer that specifies the maximum capacity of the ith box.  If
    `box_sizes[i]` equals `n` (or greater), the ith box can accommodate all `n`
    balls and thus effectively has unlimited capacity.
    ACKNOWLEDGMENT
    I'd like to thank Chris Rebert for helping me to convert my prototype
    class-based code into a generator function.
    """
    if not isinstance(balls, int):
        raise TypeError("balls must be a non-negative integer.")
    if balls < 0:
        raise ValueError("balls must be a non-negative integer.")

    if not isinstance(box_sizes, list):
        raise ValueError("box_sizes must be a non-empty list.")

    capacity = 0
    for size in box_sizes:
        if not isinstance(size, int):
            raise TypeError("box_sizes must contain only positive integers.")
        if size < 1:
            raise ValueError("box_sizes must contain only positive integers.")
        capacity += size

    if capacity < balls:
        raise ValueError(
            "The total capacity of the boxes is less than the "
            "number of balls to be distributed."
        )

    return _unlabeled_balls_in_labeled_boxes(balls, box_sizes)


def _unlabeled_balls_in_labeled_boxes(balls, box_sizes):
    """
    This recursive generator function was designed to be returned by
    `unlabeled_balls_in_labeled_boxes`.
    """

    # If there are no balls, all boxes must be empty:
    if not balls:
        yield len(box_sizes) * (0,)

    elif len(box_sizes) == 1:

        # If the single available box has sufficient capacity to store the balls,
        # there is only one possible distribution, and we return it to the caller
        # via `yield`.  Otherwise, the flow of control will pass to the end of the
        # function, triggering a `StopIteration` exception.
        if box_sizes[0] >= balls:
            yield (balls,)

    else:

        # Iterate over the number of balls in the first box (from the maximum
        # possible down to zero), recursively invoking the generator to distribute
        # the remaining balls among the remaining boxes.
        for balls_in_first_box in range(min(balls, box_sizes[0]), -1, -1):
            balls_in_other_boxes = balls - balls_in_first_box

            for distribution_other in _unlabeled_balls_in_labeled_boxes(
                balls_in_other_boxes, box_sizes[1:]
            ):
                yield (balls_in_first_box,) + distribution_other
