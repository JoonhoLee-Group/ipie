from numba import jit
import math
import numpy as np
from itertools import product

def cart2frac(reciprocal_vectors, kpts):
    """
    Convert k-points from Cartesian to fractional coordinates.
    kpts: (nkpts, 3) array of k-points in Cartesian coordinates
    cell: (3, 3) array of lattice vectors in Cartesian coordinates
    """
    b_inv = np.linalg.inv(reciprocal_vectors)
    kpts_frac = np.dot(kpts, b_inv)
    return kpts_frac

@jit(nopython=True, fastmath=True)
def BZ_to_1BZ(kpts):
    """
    Map k-points to the first Brillouin zone.
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    """
    kpts = np.where(np.abs(kpts - 0.5) < 1e-8, 0.5 - 1e-12, kpts)
    kpts = np.where(np.abs(kpts + 0.5) < 1e-8, -0.5 - 1e-12, kpts)
    kpts = np.floor(0.5 - kpts) + kpts
    return kpts

def find_translated_index(kpt, q_vec, kpts_list, tol=1e-6):
    """
    Find the index of the k-point that is translated by trs_vector for the whole k point lists
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    kpt_translated = kpt + q_vec
    fbz_kpt_trs = BZ_to_1BZ(kpt_translated)

    for j in range(len(kpts_list)):
        if np.allclose(fbz_kpt_trs, kpts_list[j], atol=tol):
            return j

@jit(nopython=True, fastmath=True)
def find_translated_index_batched(kpts, q_vec, tol=1e-6):
    """
    Find the index of the k-point that is translated by trs_vector for the whole k point lists
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    kpts_translated = kpts + q_vec
    fbz_kpts_trs = BZ_to_1BZ(kpts_translated)
    # print(fbz_kpts_trs)

    for i in range(fbz_kpts_trs.shape[0]):
        kpt = fbz_kpts_trs[i]
        for j in range(len(kpts)):
            if np.allclose(kpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = np.array(idxlis, dtype=np.int64)
    return idxlis

def find_inverted_index(kpt, kpts_list, tol=1e-6):
    """
    Find the index of the k-point that is transformed to -k for
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    mkpt = -kpt
    fbz_mkpt = BZ_to_1BZ(mkpt)
    for j in range(len(kpts_list)):
        if np.allclose(fbz_mkpt, kpts_list[j], atol=tol):
            return j

def find_inverted_index_batched(kpts, tol=1e-6):
    """
    Find the index of the k-point that is transformed to -k for
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    mkpts = -kpts
    fbz_mkpts = BZ_to_1BZ(mkpts)
    for i in range(fbz_mkpts.shape[0]):
        fbz_mkpt = fbz_mkpts[i]
        for j in range(len(kpts)):
            if np.allclose(fbz_mkpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = np.array(idxlis, dtype=np.int64)
    return idxlis


def find_gamma_pt(kpt):
    """
    Find the gamma point index
    kpt: (nk, 3) array of k-points in fractional coordinates
    """
    for i in range(kpt.shape[0]):
        if np.allclose(kpt[i], 0.0):
            return i
    
def find_self_inverse_set(kpts):
    """
    Find the set of k-points that are self-inverse
    kpts: (nk, 3) array of k-points in fractional coordinates
    """
    mq_vec = find_inverted_index_batched(kpts)
    nk = kpts.shape[0]
    self_inv_set = []
    for ik in range(nk):
        if mq_vec[ik] == ik:
            self_inv_set.append(ik)
    return np.array(self_inv_set)

def find_idx_k_mod_neg(mq):
    """
    Find the union of S and Q+ set
    """
    nk = mq.shape[0]
    smaller_indices = np.where(np.arange(nk) < mq, np.arange(nk), mq)
    unique_indices = np.unique(smaller_indices)
    return unique_indices

def find_Qplus(kpts):
    """
    Find the set of k-points that are not self-inverse mod inversion
    """
    mq_vec = find_inverted_index_batched(kpts)
    nk = kpts.shape[0]
    unique_indices = find_idx_k_mod_neg(mq_vec)
    Sset = find_self_inverse_set(kpts)
    Qplus = np.setdiff1d(unique_indices, Sset)
    return Qplus

def get_walker_from_trial(trial_wfn):
    """
    Get initial walker from trial wavefunction
    trial_wfn: numpy.ndarray with shape (nk, nbasis, nocc)
    
    Returns:
    walkers : numpy.ndarray with shape (nk, nbasis, nk, nocc)
    """
    assert len(trial_wfn.shape) == 3
    nk, nbasis, nocc = trial_wfn.shape
    walkers = np.zeros((nk, nbasis, nk, nocc), dtype=np.complex128)
    for i in range(nk):
        walkers[i, :, i, :] = trial_wfn[i]
    return walkers

def get_ni_from_idx(idx, meshsize):
    """
    Get the i-th index of the k point from the index, n3 is the fastest changing index
    """
    N1, N2, N3 = meshsize
    n1 = idx // (N2 * N3)
    n2 = (idx - n1 * N2 * N3) // N3
    n3 = idx - n1 * N2 * N3 - n2 * N3
    return n1, n2, n3

def get_possible_Gs(iq, kpts_frac, meshsize):
    pass #TODO: finish

def generate_MPmesh_3d(meshsize):
    """
    Generate the Monkhorst-Pack mesh
    For MP mesh, the k points are generated by the formula:
    k_i = (n_i + (1-N_i)/2) / N_i
    meshsize: tuple of 3 integers
    """
    N1, N2, N3 = meshsize
    kpts = []
    for n1 in range(N1):
        for n2 in range(N2):
            for n3 in range(N3):
                kpt = np.array([(n1 + 0.5 * (1 - N1)) / N1, (n2 + 0.5 * (1 - N2)) / N2, (n3 + 0.5 * (1 - N3)) / N3])
                kpts.append(kpt)
    
    kpts = np.array(kpts)
    
    # check if gamma point is in the mesh
    gamma = np.array([0.0, 0.0, 0.0])
    if not any(np.allclose(kpt, gamma) for kpt in kpts):
        kpts -= kpts[0]
        kpts = BZ_to_1BZ(kpts)
    return kpts

def get_k_from_G_MPmesh(iq, G, meshsize):
    """
    Get the list of k point indices from the q vector, G vector and the mesh size for Monkhorst-Pack mesh
    For MP mesh, the k points are generated by the formula:
    k_i = (n_i + (1-N_i)/2) / N_i

    Now only support 3D mesh, and the meshsize is a tuple of 3 odd numbers. (for even mesh size we need to shift the mesh to pass the gamma point)
    """
    ni = get_ni_from_idx(iq, meshsize)
    # ni[i] should be an integer between 0 and N_i - 1
    assert np.all(np.array(ni) < np.array(meshsize)) and np.all(np.array(ni) >= 0)
    N1, N2, N3 = meshsize
    nk = []

    for i in range(3):
        if ni[i] < meshsize[i] / 2 and ni[i] > meshsize[i] / 2  - 1:
            assert abs(G[i]) < 1e-10
            nk.append(range(meshsize[i]))
        elif ni[i] >= meshsize[i] / 2:
            assert G[i] <= 1e-10 # G[i] = 0 or -1
            ub = math.floor(1.5 * meshsize[i] - 1 - ni[i])
            if G[i] == 0:
                nk.append(range(0, ub + 1))
            else:
                nk.append(range(ub + 1, meshsize[i]))
        else:
            assert G[i] >= -1e-10 # G[i] = 0 or 1
            lb = math.ceil(0.5 * meshsize[i] - 1 - ni[i])
            if G[i] == 0:
                nk.append(range(lb, meshsize[i]))
            else:
                nk.append(range(0, lb))
    
    nkflat = [
    n3 + n2 * N3 + n1 * N2 * N3
    for n1, n2, n3 in product(nk[0], nk[1], nk[2])
    ]
    return np.array(nkflat)