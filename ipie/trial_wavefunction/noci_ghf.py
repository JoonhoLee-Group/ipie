"""Multi-determinant (NOCI) GHF trial wavefunction.

The trial is a linear combination of generalized (spinor) Slater determinants

    |Psi_T> = sum_k c_k |Phi_k>,    Phi_k : (2*nbasis, nocc) complex,

which is exactly the form produced by spin-symmetry projection of a single GHF
determinant: |Psi_T> = P |Phi> = sum_g w_g R_g |Phi>, with R_g a spin rotation
on the grid g and w_g the quadrature weight (S_z or singlet projector).

It subclasses :class:`SingleDetGHF` so that the GHF walker, the Hubbard
discrete-Hirsch propagator, and the GHF energy estimator are all selected by the
existing ``isinstance`` / dispatch machinery; the reference determinant
``self.psi0`` (used for walker initialization and reorthogonalization) is the
first determinant in the expansion.

Overlaps and Green's functions follow the ipie GHF convention
(:func:`greens_function_single_det_ghf`): for a walker phi and determinant Phi_k,

    O_k        = det(phi^T Phi_k^*) = <Phi_k|phi>      (up to the walker log_shift)
    Ghalf_k    = (phi^T Phi_k^*)^{-1} phi^T
    G_k[p, q]  = (Phi_k^* Ghalf_k)[p, q] = <c_p^dag c_q>_k

and the multi-determinant mixed quantities are the overlap-weighted averages

    S = sum_k c_k^* O_k,   G_mixed = sum_k c_k^* O_k G_k / S .
"""

import numpy

from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


class NOCIGHF(SingleDetGHF):
    def __init__(self, dets, coeffs, num_elec, num_basis, verbose: bool = False):
        dets = numpy.asarray(dets, dtype=numpy.complex128)
        assert dets.ndim == 3, "dets must have shape (ndet, 2*nbasis, nocc)"
        ndet, nrows, nocc = dets.shape
        assert nrows == 2 * num_basis
        # initialize the SingleDetGHF machinery from the reference determinant
        super().__init__(dets[0].copy(), num_elec, num_basis, verbose=verbose)
        # keep full complex determinants even if the reference happened to be real
        self.dets = dets
        self.dets_conj = dets.conj()
        self.coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
        assert self.coeffs.shape[0] == ndet
        self._num_dets = ndet
        self._max_num_dets = ndet
        # ensure the reference orbitals stay complex for the propagator buffers
        self.psi0 = dets[0].copy()
        self.psi0a = self.psi0[: self.nbasis, :]
        self.psi0b = self.psi0[self.nbasis :, :]

    @property
    def num_dets(self) -> int:
        return self._num_dets

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in NOCIGHF trial.")

    # -- per-walker, per-determinant overlaps and Green's functions ----------
    def _per_det(self, walkers):
        """Return (O, G) with O[k, w] overlaps and G[k, w] transition GFs.

        O[k, w] = <Phi_k|phi_w> (with walker log_shift), G[k,w] = (2nb,2nb) GF.
        """
        nw = walkers.nwalkers
        K = self._num_dets
        phiT = numpy.transpose(walkers.phi, (0, 2, 1))  # (w, nocc, 2nb)
        O = numpy.empty((K, nw), dtype=numpy.complex128)
        G = numpy.empty((K, nw, 2 * self.nbasis, 2 * self.nbasis), dtype=numpy.complex128)
        for k in range(K):
            dk = self.dets[k]
            dkc = self.dets_conj[k]
            # ovlp_mat[w] = phi_w^T Phi_k^* , shape (w, nocc, nocc)
            ovlp_mat = numpy.einsum("wij,jk->wik", phiT, dkc, optimize=True)
            sign, logdet = numpy.linalg.slogdet(ovlp_mat)
            O[k] = sign * numpy.exp(logdet - walkers.log_shift)
            inv = numpy.linalg.inv(ovlp_mat)
            ghalf = numpy.einsum("wij,wjk->wik", inv, phiT, optimize=True)  # (w,nocc,2nb)
            G[k] = numpy.einsum("pj,wjk->wpk", dkc, ghalf, optimize=True)   # (w,2nb,2nb)
        return O, G

    def _mixed(self, walkers):
        O, G = self._per_det(walkers)
        cstar = self.coeffs.conj()[:, None]            # (K,1)
        wts = cstar * O                                # (K, w)
        S = numpy.sum(wts, axis=0)                     # (w,)
        Gmix = numpy.einsum("kw,kwpq->wpq", wts, G, optimize=True) / S[:, None, None]
        return S, Gmix, O, G

    def calc_overlap(self, walkers) -> numpy.ndarray:
        O, _ = self._per_det(walkers)
        return numpy.sum(self.coeffs.conj()[:, None] * O, axis=0)

    def calc_greens_function(self, walkers, build_full: bool = False) -> numpy.ndarray:
        S, Gmix, _, _ = self._mixed(walkers)
        walkers.G = Gmix
        walkers.Ga = walkers.G[:, : self.nbasis, : self.nbasis]
        walkers.Gb = walkers.G[:, self.nbasis :, self.nbasis :]
        synchronize()
        return S

    # -- variational trial energy: <Psi_T|H|Psi_T> / <Psi_T|Psi_T> -----------
    def calculate_energy(self, system, hamiltonian) -> None:
        T = hamiltonian.T
        U = hamiltonian.U
        nb = self.nbasis
        K = self._num_dets
        c = self.coeffs
        num = 0.0 + 0j
        den = 0.0 + 0j
        # full double sum over bra k and ket l; transition GF <k|c^dag c|l>.
        for k in range(K):
            dkc = self.dets_conj[k]
            for l in range(K):
                dl = self.dets[l]
                A = dkc.T @ dl
                sign, logdet = numpy.linalg.slogdet(A)
                O = sign * numpy.exp(logdet)
                w = numpy.conj(c[k]) * c[l] * O
                den += w
                # G = dl inv(A) dkc^T is the transpose of the ipie-convention
                # transition GF <k|c_p^dag c_q|l>, so the one-body contraction is
                # "ij,ji" (matches the "ij,ij" used elsewhere on the un-transposed G).
                G = dl @ numpy.linalg.solve(A, dkc.T)
                Gaa, Gbb = G[:nb, :nb], G[nb:, nb:]
                Gab, Gba = G[:nb, nb:], G[nb:, :nb]
                e1 = numpy.einsum("ij,ji", T[0], Gaa) + numpy.einsum("ij,ji", T[1], Gbb)
                eU = U * numpy.sum(numpy.diag(Gaa) * numpy.diag(Gbb)
                                   - numpy.diag(Gab) * numpy.diag(Gba))
                num += w * (e1 + eU)
        self.energy = num / den
        self.e1b = None
        self.e2b = None
        if self.verbose:
            print(f"# NOCIGHF variational energy: {self.energy.real:.8f}")
