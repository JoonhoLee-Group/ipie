import time

import numpy
import scipy.linalg

from ipie.legacy.estimators.ci import (get_hmatel, get_one_body_matel,
                                       get_perm, map_orb)
from ipie.legacy.estimators.greens_function import (gab, gab_mod, gab_mod_ovlp,
                                                    gab_spin)
from ipie.legacy.estimators.local_energy import (local_energy,
                                                 variational_energy,
                                                 variational_energy_ortho_det)
from ipie.utils.io import get_input_value, write_qmcpack_wfn
from ipie.utils.mpi import get_shared_array

try:
    from ipie.lib.wicks import wicks_helper
except ImportError:
    wicks_helper = None

import numpy


class MultiSlater(object):
    def __init__(
        self,
        system,
        hamiltonian,
        wfn,
        nbasis=None,
        options={},
        init=None,
        verbose=False,
        orbs=None,
    ):
        self.verbose = verbose
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        init_time = time.time()
        self.name = "MultiSlater"
        self.type = "MultiSlater"
        # TODO : Fix for MSD.
        # This is for the overlap trial
        if len(wfn) == 3:
            # CI type expansion.
            self.from_phmsd(system.nup, system.ndown, hamiltonian.nbasis, wfn, orbs)
            self.ortho_expansion = True
        else:
            self.psi = wfn[1]
            imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
            if imag_norm <= 1e-8:
                # print("# making trial wavefunction MO coefficient real")
                self.psi = numpy.array(self.psi.real, dtype=numpy.float64)
            self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
            self.ortho_expansion = False

        self.psia = self.psi[:, :, : system.nup]
        self.psib = self.psi[:, :, system.nup :]

        self.split_trial_local_energy = get_input_value(
            options, "split_trial_local_energy", default=False, verbose=verbose
        )
        self.compute_trial_energy = get_input_value(
            options, "compute_trial_energy", default=True, verbose=verbose
        )

        # if verbose:
        # print("# compute_trial_energy = {}".format(self.compute_trial_energy))
        # print("# split_trial_local_energy = {}".format(self.split_trial_local_energy))

        if self.split_trial_local_energy:
            if verbose:
                print(
                    "# taking the determinant with the largest coefficient as the local energy trial"
                )
            imax = numpy.argmax(numpy.abs(self.coeffs))
            self.le_coeffs = numpy.array([self.coeffs[imax]], dtype=numpy.complex128)
            self.le_psi = numpy.array([self.psi[imax, :, :]], dtype=self.psi.dtype)
            self.le_ortho_expansion = self.ortho_expansion
        else:
            self.le_psi = self.psi.copy()
            self.le_coeffs = self.coeffs.copy()
            self.le_ortho_expansion = self.ortho_expansion

        if self.verbose:
            if self.ortho_expansion:
                print("# Assuming orthogonal trial wavefunction expansion.")
            else:
                print("# Assuming non-orthogonal trial wavefunction expansion.")
            print("# Trial wavefunction shape: {}".format(self.psi.shape))

        self.ndets = get_input_value(
            options, "ndets", default=len(self.coeffs), verbose=verbose
        )
        # if self.verbose:
        # print("# Setting ndets: {}".format(self.ndets))

        if self.ndets == 1:
            self.G, self.Ghalf = gab_spin(
                self.psi[0], self.psi[0], system.nup, system.ndown
            )
            self.G = numpy.array(self.G, dtype=numpy.complex128)
            self.Ghalf = [
                numpy.array(self.Ghalf[0], dtype=numpy.complex128),
                numpy.array(self.Ghalf[1], dtype=numpy.complex128),
            ]
        else:
            self.G = None
            self.Ghalf = None
        if init is not None:
            if verbose:
                print("# Using initial wavefunction from file.")
            self.init = init
        else:
            if verbose:
                print(
                    "# Setting initial wavefunction as first determinant in"
                    " expansion."
                )
            if len(self.psi.shape) == 3:
                self.init = self.psi[0].copy()
            else:
                self.init = self.psi.copy()

        self.wicks = get_input_value(options, "wicks", default=False, verbose=verbose)
        self.use_wicks_helper = get_input_value(
            options, "use_wicks_helper", default=False, verbose=verbose
        )
        self.optimized = get_input_value(
            options, "optimized", default=False, verbose=verbose
        )
        if self.wicks:  # this is for Wick's theorem
            # if True: # this is for Wick's theorem
            if verbose:
                print("# Using generalized Wick's theorem for the PHMSD trial")
                print(
                    "# Setting the first determinant in"
                    " expansion as the reference wfn for Wick's theorem."
                )
            self.psi0a = self.psi[0, :, : system.nup].copy()
            self.psi0b = self.psi[0, :, system.nup :].copy()
            if verbose:
                print("# Setting additional member variables for Wick's theorem")
            d0a = self.occa[0]
            d0b = self.occb[0]
            self.cre_a = [
                []
            ]  # one empty list as a member to account for the reference state
            self.anh_a = [
                []
            ]  # one empty list as a member to account for the reference state
            self.cre_b = [
                []
            ]  # one empty list as a member to account for the reference state
            self.anh_b = [
                []
            ]  # one empty list as a member to account for the reference state
            self.phase_a = numpy.ones(self.ndets)  # 1.0 is for the reference state
            self.phase_b = numpy.ones(self.ndets)  # 1.0 is for the reference state
            nexcit_a = system.nup
            nexcit_b = system.ndown
            # This is an overestimate because we don't know number of active
            # electrons in trial from read in.
            # TODO work this out.
            max_excit = max(nexcit_a, nexcit_b) + 1
            cre_ex_a = [[] for _ in range(max_excit)]
            cre_ex_b = [[] for _ in range(max_excit)]
            anh_ex_a = [[] for _ in range(max_excit)]
            anh_ex_b = [[] for _ in range(max_excit)]
            excit_map_a = [[] for _ in range(max_excit)]
            excit_map_b = [[] for _ in range(max_excit)]
            for j in range(1, self.ndets):
                dja = self.occa[j]
                djb = self.occb[j]

                anh_a = list(
                    set(dja) - set(d0a)
                )  # annihilation to right, creation to left
                cre_a = list(
                    set(d0a) - set(dja)
                )  # creation to right, annhilation to left

                anh_b = list(set(djb) - set(d0b))
                cre_b = list(set(d0b) - set(djb))

                cre_a.sort()
                cre_b.sort()
                anh_a.sort()
                anh_b.sort()

                # anh_a = numpy.array(anh_a)
                # cre_a = numpy.array(cre_a)
                # anh_b = numpy.array(anh_b)
                # cre_b = numpy.array(cre_b)

                self.anh_a += [anh_a]
                self.anh_b += [anh_b]
                self.cre_a += [cre_a]
                self.cre_b += [cre_b]
                anh_ex_a[len(anh_a)].append(anh_a)
                anh_ex_b[len(anh_b)].append(anh_b)
                cre_ex_a[len(cre_a)].append(cre_a)
                cre_ex_b[len(cre_b)].append(cre_b)
                excit_map_a[len(anh_a)].append(j)
                excit_map_b[len(anh_b)].append(j)

                perm_a = get_perm(anh_a, cre_a, d0a, dja)
                perm_b = get_perm(anh_b, cre_b, d0b, djb)

                if perm_a:
                    self.phase_a[j] = -1
                else:
                    self.phase_a[j] = +1

                if perm_b:
                    self.phase_b[j] = -1
                else:
                    self.phase_b[j] = +1

            self.ndet_a = [len(ex) for ex in cre_ex_a]
            self.ndet_b = [len(ex) for ex in cre_ex_b]
            self.max_excite_a = max(
                -1 if nd == 0 else i for i, nd in enumerate(self.ndet_a)
            )
            self.max_excite_b = max(
                -1 if nd == 0 else i for i, nd in enumerate(self.ndet_b)
            )
            self.max_excite = max(self.max_excite_a, self.max_excite_b)
            self.cre_ex_a = [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_a]
            self.cre_ex_b = [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_b]
            self.anh_ex_a = [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_a]
            self.anh_ex_b = [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_b]
            self.excit_map_a = [
                numpy.array(ex, dtype=numpy.int32) for ex in excit_map_a
            ]
            self.excit_map_b = [
                numpy.array(ex, dtype=numpy.int32) for ex in excit_map_b
            ]
        self.compute_opdm = options.get("compute_opdm", True)
        if self.ortho_expansion and self.compute_opdm:  # this is for phmsd
            if verbose:
                print("# Computing 1-RDM of the trial wfn for mean-field shift")
            start = time.time()
            if self.use_wicks_helper:
                assert wicks_helper is not None, "wicks_helper lib not found."
                dets = wicks_helper.encode_dets(self.occa, self.occb)
                phases = wicks_helper.convert_phase(self.occa, self.occb)
                assert numpy.max(numpy.abs(self.coeffs.imag)) < 1e-12
                self.G = wicks_helper.compute_opdm(
                    phases * self.coeffs.real.copy(),
                    dets,
                    hamiltonian.nbasis,
                    system.ne,
                )
            else:
                self.G = self.compute_1rdm(hamiltonian.nbasis)
            end = time.time()
            if verbose:
                print("# Time to compute 1-RDM: {} s".format(end - start))

        self.error = False
        self.initialisation_time = time.time() - init_time
        self._nalpha = system.nup
        self._nbeta = system.ndown
        self._nelec = system.nelec
        self._nbasis = hamiltonian.nbasis
        self._rchol = None
        self._rchola = None
        self._rcholb = None
        self._UVT = None
        self._eri = None
        self._mem_required = 0.0
        self.ecoul0 = None
        self.exxa0 = None
        self.exxb0 = None
        write_wfn = options.get("write_wavefunction", False)
        output_file = options.get("output_file", "wfn.h5")

        if write_wfn:
            self.write_wavefunction(filename=output_file)
        if verbose:
            print("# Finished setting up trial_wavefunction.MultiSlater.")

    def local_energy_2body(self, system, hamiltonian):
        """Compute walkers two-body local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """

        nalpha, nbeta = system.nup, system.ndown
        nbasis = hamiltonian.nbasis
        naux = self._rchol.shape[1]

        Ga, Gb = self.Ghalf[0], self.Ghalf[1]
        Xa = self._rchol[: nalpha * nbasis].T.dot(Ga.ravel())
        Xb = self._rchol[nalpha * nbasis :].T.dot(Gb.ravel())
        ecoul = numpy.dot(Xa, Xa)
        ecoul += numpy.dot(Xb, Xb)
        ecoul += 2 * numpy.dot(Xa, Xb)
        rchol_a, rchol_b = (
            self._rchol[: nalpha * nbasis],
            self._rchol[nalpha * nbasis :],
        )

        rchol_a = rchol_a.T
        rchol_b = rchol_b.T
        Ta = numpy.zeros((naux, nalpha, nalpha), dtype=rchol_a.dtype)
        Tb = numpy.zeros((naux, nbeta, nbeta), dtype=rchol_b.dtype)
        GaT = Ga.T
        GbT = Gb.T
        for x in range(naux):
            rmi_a = rchol_a[x].reshape((nalpha, nbasis))
            Ta[x] = rmi_a.dot(GaT)
            rmi_b = rchol_b[x].reshape((nbeta, nbasis))
            Tb[x] = rmi_b.dot(GbT)
        exxa = numpy.tensordot(Ta, Ta, axes=((0, 1, 2), (0, 2, 1)))
        exxb = numpy.tensordot(Tb, Tb, axes=((0, 1, 2), (0, 2, 1)))

        exx = exxa + exxb
        e2b = 0.5 * (ecoul - exx)

        return ecoul, exxa, exxb

    def calculate_energy(self, system, hamiltonian):
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        # Cannot use usual energy evaluation routines if trial is orthogonal.
        if self.ortho_expansion:
            self.energy, self.e1b, self.e2b = variational_energy_ortho_det(
                system, hamiltonian, self.spin_occs, self.coeffs
            )
        else:
            (self.energy, self.e1b, self.e2b) = variational_energy(
                system, hamiltonian, self
            )
        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )
            print("# Time to evaluate local energy: {} s".format(time.time() - start))

    def from_phmsd(self, nup, ndown, nbasis, wfn, orbs):
        ndets = len(wfn[0])
        ne = nup + ndown
        self.psi = numpy.zeros((ndets, nbasis, ne), dtype=numpy.complex128)
        if self.verbose:
            print("# Creating trial wavefunction from CI-like expansion.")
        if orbs is None:
            if self.verbose:
                print("# Assuming RHF reference.")
            I = numpy.eye(nbasis, dtype=numpy.complex128)
        # Store alpha electrons first followed by beta electrons.
        nb = nbasis
        dets = [list(a) + [i + nb for i in c] for (a, c) in zip(wfn[1], wfn[2])]
        self.spin_occs = [numpy.sort(d) for d in dets]
        self.occa = numpy.array(wfn[1])
        self.occb = numpy.array(wfn[2])
        self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
        for idet, (occa, occb) in enumerate(zip(wfn[1], wfn[2])):
            self.psi[idet, :, :nup] = I[:, occa]
            self.psi[idet, :, nup:] = I[:, occb]

    def recompute_ci_coeffs(self, nup, ndown, ham):
        H = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        S = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        m = ham.nbasis
        na = nup
        nb = ndown
        if self.ortho_expansion:
            for i in range(self.ndets):
                for j in range(i, self.ndets):
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    H[i, j] = get_hmatel(ham, nup + ndown, di, dj)[0]
            e, ev = scipy.linalg.eigh(H, lower=False)
        else:
            na = nup
            for i, di in enumerate(self.psi):
                for j, dj in enumerate(self.psi):
                    if j >= i:
                        ga, gha, ioa = gab_mod_ovlp(di[:, :na], dj[:, :na])
                        gb, ghb, iob = gab_mod_ovlp(di[:, na:], dj[:, na:])
                        G = numpy.array([ga, gb])
                        Ghalf = numpy.array([gha, ghb])
                        ovlp = 1.0 / (scipy.linalg.det(ioa) * scipy.linalg.det(iob))
                        if abs(ovlp) > 1e-12:
                            if self._rchol is not None:
                                rchol = self.rchol(i)
                            else:
                                rchol = None
                            H[i, j] = (
                                ovlp * local_energy(ham, G, Ghalf=Ghalf, rchol=rchol)[0]
                            )
                            S[i, j] = ovlp
                            H[j, i] = numpy.conjugate(H[i, j])
                            S[j, i] = numpy.conjugate(S[i, j])
            e, ev = scipy.linalg.eigh(H, S, lower=False)
        # if self.verbose:
        # print("Old and New CI coefficients: ")
        # for co,cn in zip(self.coeffs,ev[:,0]):
        # print("{} {}".format(co, cn))
        return numpy.array(ev[:, 0], dtype=numpy.complex128)

    def compute_1rdm(self, nbasis):
        assert self.ortho_expansion == True
        denom = numpy.sum(self.coeffs.conj() * self.coeffs)
        Pa = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
        Pb = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
        P = [Pa, Pb]
        for idet in range(self.ndets):
            di = self.spin_occs[idet]
            # zero excitation case
            for iorb in range(len(di)):
                ii, spin_ii = map_orb(di[iorb], nbasis)
                P[spin_ii][ii, ii] += self.coeffs[idet].conj() * self.coeffs[idet]
            for jdet in range(idet + 1, self.ndets):
                dj = self.spin_occs[jdet]
                from_orb = list(set(dj) - set(di))
                to_orb = list(set(di) - set(dj))
                nex = len(from_orb)
                if nex > 1:
                    continue
                elif nex == 1:
                    perm = get_perm(from_orb, to_orb, di, dj)
                    if perm:
                        phase = -1
                    else:
                        phase = 1
                    ii, si = map_orb(from_orb[0], nbasis)
                    aa, sa = map_orb(to_orb[0], nbasis)
                    if si == sa:
                        P[si][aa, ii] += (
                            self.coeffs[jdet].conj() * self.coeffs[idet] * phase
                        )
                        P[si][ii, aa] += (
                            self.coeffs[jdet] * self.coeffs[idet].conj() * phase
                        )
        P[0] /= denom
        P[1] /= denom

        return P

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        import cupy

        size = self.coeffs.size
        if numpy.isrealobj(self.psi):
            size += self.psi.size / 2.0
        else:
            size += self.psi.size
        if numpy.isrealobj(self._rchola):
            size += self._rchola.size / 2.0 + self._rcholb.size / 2.0
        else:
            size += self._rchola.size + self._rcholb.size
        if type(self.G) == numpy.ndarray:
            size += self.G.size
        if self.Ghalf != None:
            size += self.Ghalf[0].size + self.Ghalf[1].size
        if self.ortho_expansion:
            size += (
                self.occa.size + self.occb.size
            ) / 2.0  # to account for the fact that these are float64, not complex128
        if verbose:
            expected_bytes = size * 16.0
            print(
                "# trial_wavefunction.MultiSlater: expected to allocate {:4.3f} GB".format(
                    expected_bytes / 1024**3
                )
            )

        self.psi = cupy.asarray(self.psi)
        self.coeffs = cupy.asarray(self.coeffs)
        self._rchola = cupy.asarray(self._rchola)
        self._rcholb = cupy.asarray(self._rcholb)
        if type(self.G) == numpy.ndarray:
            self.G = cupy.asarray(self.G)
        if self.Ghalf != None:
            self.Ghalf[0] = cupy.asarray(self.Ghalf[0])
            self.Ghalf[1] = cupy.asarray(self.Ghalf[1])
        if self.ortho_expansion:
            self.occa = cupy.asarray(self.occa)
            self.occb = cupy.asarray(self.occb)
        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print(
                "# trial_wavefunction.MultiSlater: using {:4.3f} GB out of {:4.3f} GB memory on GPU".format(
                    used_bytes / 1024**3, total_bytes / 1024**3
                )
            )

    def contract_one_body(self, ints):
        numer = 0.0
        denom = 0.0
        na = self._nalpha
        for i in range(self.ndets):
            for j in range(self.ndets):
                cfac = self.coeffs[i].conj() * self.coeffs[j]
                if self.ortho_expansion:
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    tij = get_one_body_matel(ints, di, dj)
                    numer += cfac * tij
                    if i == j:
                        denom += self.coeffs[i].conj() * self.coeffs[i]
                else:
                    di = self.psi[i]
                    dj = self.psi[j]
                    ga, gha, ioa = gab_mod_ovlp(di[:, :na], dj[:, :na])
                    gb, ghb, iob = gab_mod_ovlp(di[:, na:], dj[:, na:])
                    ovlp = 1.0 / (scipy.linalg.det(ioa) * scipy.linalg.det(iob))
                    tij = numpy.dot(ints.ravel(), ga.ravel() + gb.ravel())
                    numer += cfac * ovlp * tij
                    denom += cfac * ovlp
        return numer / denom

    def write_wavefunction(self, filename="wfn.h5", init=None, occs=False):
        if occs:
            wfn = (self.coeffs, self.occa, self.occb)
        else:
            wfn = (self.coeffs, self.psi)
        write_qmcpack_wfn(filename, wfn, "uhf", self._nelec, self._nbasis, init=init)

    def half_rotate(self, system, hamiltonian, comm=None):
        # Half rotated cholesky vectors (by trial wavefunction or a reference wfn in the case of PHMSD).
        na = system.nup
        nb = system.ndown
        M = hamiltonian.nbasis
        nchol = hamiltonian.chol_vecs.shape[-1]
        if self.verbose:
            print("# Constructing half rotated Cholesky vectors.")

        hr_ndet = self.ndets
        if self.ortho_expansion:
            if self.verbose:
                print("# Only performing half-rotation of the reference determinant")
            hr_ndet = 1

        if isinstance(hamiltonian.chol_vecs, numpy.ndarray):
            chol = hamiltonian.chol_vecs.reshape((M, M, nchol))
        else:
            chol = hamiltonian.chol_vecs.toarray().reshape((nchol, M, M))

        if hamiltonian.exact_eri:
            shape = (hr_ndet, (M**2 * (na**2 + nb**2) + M**2 * (na * nb)))
            self._eri = get_shared_array(comm, shape, numpy.complex128)
            self._mem_required = self._eri.nbytes / (1024.0**3.0)

            for i, psi in enumerate(self.psi[:hr_ndet]):
                vipjq_aa = numpy.einsum(
                    "Xmp,Xrq,mi,rj->ipjq",
                    chol,
                    chol,
                    psi[:, :na].conj(),
                    psi[:, :na].conj(),
                    optimize=True,
                )
                vipjq_bb = numpy.einsum(
                    "Xmp,Xrq,mi,rj->ipjq",
                    chol,
                    chol,
                    psi[:, na:].conj(),
                    psi[:, na:].conj(),
                    optimize=True,
                )
                vipjq_ab = numpy.einsum(
                    "Xmp,Xrq,mi,rj->ipjq",
                    chol,
                    chol,
                    psi[:, :na].conj(),
                    psi[:, na:].conj(),
                    optimize=True,
                )
                self._eri[i, : M**2 * na**2] = vipjq_aa.ravel()
                self._eri[
                    i, M**2 * na**2 : M**2 * na**2 + M**2 * nb**2
                ] = vipjq_bb.ravel()
                self._eri[i, M**2 * na**2 + M**2 * nb**2 :] = vipjq_ab.ravel()

                if hamiltonian.pno:
                    thresh_pno = hamiltonian.thresh_pno
                    UVT_aa = []
                    UVT_bb = []
                    UVT_ab = []

                    nocca = system.nup
                    noccb = system.ndown
                    nvira = hamiltonian.nbasis - system.nup
                    nvirb = hamiltonian.nbasis - system.ndown

                    r_aa = []
                    for i in range(na):
                        for j in range(i, na):
                            Vab = vipjq_aa[i, :, j, :]
                            U, s, VT = numpy.linalg.svd(Vab)
                            idx = s > thresh_pno
                            U = U[:, idx]
                            s = s[idx]
                            r_aa += [s.shape[0] / float(hamiltonian.nbasis)]
                            VT = VT[idx, :]
                            U = U.dot(numpy.diag(numpy.sqrt(s)))
                            VT = numpy.diag(numpy.sqrt(s)).dot(VT)
                            UVT_aa += [(U, VT)]
                    r_aa = numpy.array(r_aa)
                    r_aa = numpy.mean(r_aa)

                    r_bb = []
                    for i in range(nb):
                        for j in range(i, nb):
                            Vab = vipjq_bb[i, :, j, :]
                            U, s, VT = numpy.linalg.svd(Vab)
                            idx = s > thresh_pno
                            U = U[:, idx]
                            s = s[idx]
                            r_bb += [s.shape[0] / float(hamiltonian.nbasis)]
                            VT = VT[idx, :]
                            U = U.dot(numpy.diag(numpy.sqrt(s)))
                            VT = numpy.diag(numpy.sqrt(s)).dot(VT)

                            UVT_bb += [(U, VT)]

                    r_bb = numpy.array(r_bb)
                    r_bb = numpy.mean(r_bb)

                    r_ab = []
                    for i in range(na):
                        for j in range(nb):
                            Vab = vipjq_ab[i, :, j, :]
                            U, s, VT = numpy.linalg.svd(Vab)
                            idx = s > thresh_pno
                            U = U[:, idx]
                            s = s[idx]
                            r_ab += [s.shape[0] / float(hamiltonian.nbasis)]
                            VT = VT[idx, :]
                            U = U.dot(numpy.diag(numpy.sqrt(s)))
                            VT = numpy.diag(numpy.sqrt(s)).dot(VT)

                            UVT_ab += [(U, VT)]

                    r_ab = numpy.array(r_ab)
                    r_ab = numpy.mean(r_ab)

                    self._UVT = [UVT_aa, UVT_bb, UVT_ab]
                    self._eri = None
                    if self.verbose:
                        print(
                            "# Average number of orbitals (relative to total) for aa, bb, ab = {}, {}, {}".format(
                                r_aa, r_bb, r_ab
                            )
                        )

            if self.verbose:
                print(
                    "# Memory required by exact ERIs: "
                    " {:.4f} GB.".format(self._mem_required)
                )
            if comm is not None:
                comm.barrier()
        # else:
        # shape = (hr_ndet*(M*(na+nb)), nchol)
        # self._rchol = get_shared_array(comm, shape, numpy.complex128)
        shape_a = (nchol, hr_ndet * (M * (na)))
        shape_b = (nchol, hr_ndet * (M * (nb)))
        self._rchola = get_shared_array(comm, shape_a, self.psi.dtype)
        self._rcholb = get_shared_array(comm, shape_b, self.psi.dtype)
        for i, psi in enumerate(self.psi[:hr_ndet]):
            start_time = time.time()
            if self.verbose:
                print(
                    "# Rotating Cholesky for determinant {} of "
                    "{}.".format(i + 1, hr_ndet)
                )
            # start = i*M*(na+nb)
            start_a = i * M * na  # determinant loops
            start_b = i * M * nb  # determinant loops
            compute = True
            # Distribute amongst MPI tasks on this node.
            if comm is not None:
                nwork_per_thread = hamiltonian.nchol // comm.size
                if nwork_per_thread == 0:
                    start_n = 0
                    end_n = nchol
                    if comm.rank != 0:
                        # Just run on root processor if problem too small.
                        compute = False
                else:
                    start_n = comm.rank * nwork_per_thread  # Cholesky work split
                    end_n = (comm.rank + 1) * nwork_per_thread
                    if comm.rank == comm.size - 1:
                        end_n = nchol
            else:
                start_n = 0
                end_n = hamiltonian.nchol

            nchol_loc = end_n - start_n
            # if comm.rank == 0:
            # print(start_n, end_n, nchol_loc)
            # print(numpy.may_share_memory(chol, chol[:,start_n:end_n]))
            if compute:
                # Investigate whether these einsums are fast in the future
                rup = numpy.einsum(
                    "mi,mnx->xin",
                    psi[:, :na].conj(),
                    chol[:, :, start_n:end_n],
                    optimize=True,
                )
                rup = rup.reshape((nchol_loc, na * M))
                rdn = numpy.einsum(
                    "mi,mnx->xin",
                    psi[:, na:].conj(),
                    chol[:, :, start_n:end_n],
                    optimize=True,
                )
                rdn = rdn.reshape((nchol_loc, nb * M))
                self._rchola[start_n:end_n, start_a : start_a + M * na] = rup[:]
                self._rcholb[start_n:end_n, start_b : start_b + M * nb] = rdn[:]

            self._mem_required = (self._rchola.nbytes + self._rcholb.nbytes) / (
                1024.0**3.0
            )
            if self.verbose:
                print(
                    "# Memory required by half-rotated integrals: "
                    " {:.4f} GB.".format(self._mem_required)
                )
                print(
                    "# Time to half-rotated integrals: {} s.".format(
                        time.time() - start_time
                    )
                )
        if comm is not None:
            comm.barrier()

        if hamiltonian.control_variate:
            self.ecoul0, self.exxa0, self.exxb0 = self.local_energy_2body(
                system, hamiltonian
            )

    def rot_chol(self, idet=0, spin=None):
        """Helper function"""
        if spin is None:
            print("# rot_chol no longer supported without spin ")
            exit()
            stride = self._nbasis * (self._nalpha + self._nbeta)
            return self._rchol[idet * stride : (idet + 1) * stride]
        else:
            # stride = self._nbasis * (self._nalpha + self._nbeta)
            if spin == 0:
                alpha = self._nbasis * self._nalpha
                stride = self._nbasis * self._nalpha
                return self._rchola[:, idet * stride : idet * stride + alpha]
            else:
                beta = self._nbasis * self._nbeta
                stride = self._nbasis * self._nbeta
                # return self._rchol[idet*stride+alpha:idet*stride+alpha+beta]
                return self._rcholb[:, idet * stride : idet * stride + beta]
