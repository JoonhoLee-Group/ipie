import h5py
import numpy

try:
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import sys

import ipie.legacy.propagation.hubbard
from ipie.estimators.utils import H5EstimatorHelper
from ipie.legacy.estimators.ekt import ekt_1h_fock_opt, ekt_1p_fock_opt
from ipie.legacy.estimators.greens_function import gab
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.propagation.generic import back_propagate_generic
from ipie.legacy.propagation.planewave import back_propagate_planewave


class BackPropagation(object):
    """Class for computing back propagated estimates.

    Parameters
    ----------
    bp : dict
        Input options for BP estimates.
    root : bool
        True if on root/master processor.
    h5f : :class:`h5py.File`
        Output file object.
    qmc : :class:`ipie.state.QMCOpts` object.
        Container for qmc input options.
    system : system object
        System object.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.
    BT2 : :class:`numpy.ndarray`
        One-body propagator for back propagation.

    Attributes
    ----------
    nmax : int
        Max number of measurements.
    header : int
        Output header.
    rdm : bool
        True if output BP RDM to file.
    nreg : int
        Number of regular estimates (exluding iteration).
    G : :class:`numpy.ndarray`
        One-particle RDM.
    estimates : :class:`numpy.ndarray`
        Store for mixed estimates per processor.
    global_estimates : :class:`numpy.ndarray`
        Store for mixed estimates accross all processors.
    output : :class:`ipie.estimators.H5EstimatorHelper`
        Class for outputting data to HDF5 group.
    rdm_output : :class:`ipie.estimators.H5EstimatorHelper`
        Class for outputting rdm data to HDF5 group.
    """

    def __init__(self, bp, root, filename, qmc, system, trial, dtype, BT2):
        self.tau_bp = bp.get("tau_bp", 0)
        self.nmax = int(self.tau_bp / qmc.dt)
        self.header = ["E", "E1b", "E2b"]
        self.calc_one_rdm = bp.get("one_rdm", True)
        self.calc_two_rdm = bp.get("two_rdm", None)
        self.init_walker = bp.get("init_walker", False)
        self.nsplit = bp.get("nsplit", 1)
        self.splits = numpy.array(
            [(i + 1) * (self.nmax // self.nsplit) for i in range(self.nsplit)]
        )
        self.nreg = len(self.header)
        self.accumulated = False
        self.eval_energy = bp.get("evaluate_energy", False)
        self.eval_ekt = bp.get("evaluate_ekt", False)
        self.G = numpy.zeros(trial.G.shape, dtype=numpy.complex128)
        self.nstblz = qmc.nstblz
        self.BT2 = BT2
        self.restore_weights = bp.get("restore_weights", None)
        if root:
            print("# restore_weights = {}".format(self.restore_weights))
        self.dt = qmc.dt
        dms_size = self.G.size
        # Abuse of language for the moment. Only accumulates S(k) for UEG.
        # TODO: Add functionality to accumulate 2RDM?
        if self.calc_two_rdm is not None:
            if self.calc_two_rdm == "structure_factor":
                two_rdm_shape = (
                    2,
                    2,
                    len(system.qvecs),
                )
            else:
                two_rdm_shape = (
                    system.nbasis,
                    system.nbasis,
                    system.nbasis,
                    system.nbasis,
                )
            self.two_rdm = numpy.zeros(two_rdm_shape, dtype=numpy.complex128)
            dms_size += self.two_rdm.size
        else:
            self.two_rdm = None

        if self.eval_ekt:
            self.ekt_fock_1h = numpy.zeros_like(self.G[0])
            self.ekt_fock_1p = numpy.zeros_like(self.G[0])
            dms_size += self.ekt_fock_1h.size
            dms_size += self.ekt_fock_1p.size

        self.estimates = numpy.zeros(self.nreg + 1 + dms_size, dtype=dtype)
        self.global_estimates = numpy.zeros(self.nreg + 1 + dms_size, dtype=dtype)
        self.key = {
            "ETotal": "BP estimate for total energy.",
            "E1B": "BP estimate for one-body energy.",
            "E2B": "BP estimate for two-body energy.",
        }
        if root:
            self.setup_output(filename)

        if trial.type == "GHF":
            self.update = self.update_ghf
            self.back_propagate = ipie.legacy.propagation.hubbard.back_propagate_ghf
        else:
            self.update = self.update_uhf
            if system.name == "Generic":
                self.back_propagate = back_propagate_generic
            elif system.name == "UEG":
                self.back_propagate = back_propagate_planewave
            else:
                self.back_propagate = ipie.legacy.propagation.hubbard.back_propagate

    def update_uhf(
        self, qmc, system, hamiltonian, trial, psi, step, free_projection=False
    ):
        """Calculate back-propagated estimates for RHF/UHF walkers.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        qmc : :class:`ipie.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        buff_ix = psi.walkers[0].field_configs.step
        if buff_ix not in self.splits:
            return
        nup = system.nup
        for i, wnm in enumerate(psi.walkers):
            if self.init_walker:
                phi_bp = numpy.array(trial.init.copy(), dtype=numpy.complex128)
            else:
                phi_bp = numpy.array(trial.psi.copy(), dtype=numpy.complex128)
            # TODO: Fix for ITCF.
            self.back_propagate(
                phi_bp,
                wnm.field_configs,
                system,
                hamiltonian,
                self.nstblz,
                self.BT2,
                self.dt,
            )
            self.G[0] = gab(phi_bp[:, :nup], wnm.phi_old[:, :nup]).T
            self.G[1] = gab(phi_bp[:, nup:], wnm.phi_old[:, nup:]).T

            if self.eval_energy:
                eloc = local_energy(system, self.G, opt=False, two_rdm=self.two_rdm)
                energies = numpy.array(list(eloc))
            else:
                energies = numpy.zeros(3)

            if (
                self.calc_two_rdm is not None
                and self.calc_two_rdm is not "structure_factor"
            ):
                # <p^+ q^+ s r> = G(p, r, q, s) also spin-summed
                self.two_rdm = numpy.einsum(
                    "pr,qs->prqs", self.G[0], self.G[0], optimize=True
                ) - numpy.einsum("ps,qr->prqs", self.G[0], self.G[0], optimize=True)
                self.two_rdm += numpy.einsum(
                    "pr,qs->prqs", self.G[1], self.G[1], optimize=True
                ) - numpy.einsum("ps,qr->prqs", self.G[1], self.G[1], optimize=True)
                self.two_rdm += numpy.einsum(
                    "pr,qs->prqs", self.G[0], self.G[1], optimize=True
                ) + numpy.einsum("pr,qs->prqs", self.G[1], self.G[0], optimize=True)

            if self.eval_ekt:
                if system.name == "UEG":
                    # there needs to be a factor of 2.0 here to account for the convention of cholesky vectors in the system class
                    chol_vecs = 2.0 * system.chol_vecs.toarray().T.reshape(
                        (system.nchol, system.nbasis, system.nbasis)
                    )
                    self.ekt_fock_1p = ekt_1p_fock_opt(
                        system.H1[0], chol_vecs, self.G[0], self.G[1]
                    )
                    self.ekt_fock_1h = ekt_1h_fock_opt(
                        system.H1[0], chol_vecs, self.G[0], self.G[1]
                    )
                else:
                    self.ekt_fock_1p = ekt_1p_fock_opt(
                        system.H1[0], system.chol_vecs, self.G[0], self.G[1]
                    )
                    self.ekt_fock_1h = ekt_1h_fock_opt(
                        system.H1[0], system.chol_vecs, self.G[0], self.G[1]
                    )

            if self.restore_weights is not None:
                cosine_fac, ph_fac = wnm.field_configs.get_wfac()
                if self.restore_weights == "full":
                    # BP-Pres
                    wfac = ph_fac / cosine_fac
                else:
                    # BP-PRes (partial)
                    wfac = ph_fac
                weight = wnm.weight * wfac
            else:
                # BP-PhL
                weight = wnm.weight

            self.estimates[: self.nreg] += weight * energies
            self.estimates[self.nreg] += weight

            start = self.nreg + 1
            end = start + self.G.size
            self.estimates[start:end] += weight * self.G.flatten()

            if self.calc_two_rdm is not None:
                start = end
                end = end + self.two_rdm.size
                self.estimates[start:end] += weight * self.two_rdm.flatten()

            if self.eval_ekt:
                start = end
                end = end + self.ekt_fock_1p.size
                self.estimates[start:end] += weight * self.ekt_fock_1p.flatten()
                start = end
                end = end + self.ekt_fock_1h.size
                self.estimates[start:end] += weight * self.ekt_fock_1h.flatten()

            if buff_ix == self.splits[-1]:
                wnm.field_configs.reset()
        if buff_ix == self.splits[-1]:
            psi.copy_historic_wfn()
        self.accumulated = True
        self.buff_ix = buff_ix

    def update_ghf(
        self, qmc, system, hamiltonian, trial, psi, step, free_projection=False
    ):
        """Calculate back-propagated estimates for GHF walkers.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        qmc : :class:`ipie.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        if step % self.nmax != 0:
            return
        print(" ***** Back Propagation with GHF is broken.")
        sys.exit()
        psi_bp = self.back_propagate(
            system, psi.walkers, trial, self.nstblz, self.BT2, self.dt
        )
        denominator = sum(wnm.weight for wnm in psi.walkers)
        nup = system.nup
        for i, (wnm, wb) in enumerate(zip(psi.walkers, psi_bp)):
            construct_multi_ghf_gab(wb.phi, wnm.phi_old, wb.weights, wb.Gi, wb.ots)
            # note that we are abusing the weights variable from the multighf
            # walker to store the reorthogonalisation factors.
            weights = wb.weights * trial.coeffs * wb.ots
            denom = sum(weights)
            energies = numpy.array(
                list(local_energy_ghf(system, wb.Gi, weights, denom))
            )
            self.G = numpy.einsum("i,ijk->jk", weights, wb.Gi) / denom
            self.estimates[1:] = self.estimates[1:] + wnm.weight * numpy.append(
                energies, self.G.flatten()
            )
        self.estimates[0] += denominator
        psi.copy_historic_wfn()
        psi.copy_bp_wfn(psi_bp)

    def print_step(self, comm, nprocs, step, nsteps=1, free_projection=False):
        """Print back-propagated estimates to file.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        if not self.accumulated:
            return
        comm.Reduce(self.estimates, self.global_estimates, op=mpi_sum)
        if comm.rank == 0:
            weight = self.global_estimates[self.nreg]
            self.output.push(
                numpy.array([weight]), "denominator_{:d}".format(self.buff_ix)
            )
            if self.eval_energy:
                if free_projection:
                    self.output.push(
                        self.global_estimates[: self.nreg],
                        "energies_{:d}".format(self.buff_ix),
                    )
                else:
                    self.output.push(
                        self.global_estimates[: self.nreg] / weight,
                        "energies_{:d}".format(self.buff_ix),
                    )
            if self.calc_one_rdm:
                start = self.nreg + 1
                end = self.nreg + 1 + self.G.size
                rdm = self.global_estimates[start:end].reshape(self.G.shape)
                self.output.push(rdm, "one_rdm_{:d}".format(self.buff_ix))
            if self.calc_two_rdm:
                start = self.nreg + 1 + self.G.size
                end = start + self.two_rdm.size
                rdm = self.global_estimates[start:end].reshape(self.two_rdm.shape)
                self.output.push(rdm, "two_rdm_{:d}".format(self.buff_ix))

            if self.eval_ekt:
                start = self.nreg + 1
                if self.calc_one_rdm:
                    start += self.G.size
                if self.calc_two_rdm:
                    start += self.two_rdm.size

                end = start + self.ekt_fock_1p.size
                fock = self.global_estimates[start:end].reshape(self.ekt_fock_1p.shape)
                self.output.push(fock, "fock_1p_{:d}".format(self.buff_ix))
                start = end
                end = end + self.ekt_fock_1h.size
                fock = self.global_estimates[start:end].reshape(self.ekt_fock_1h.shape)
                self.output.push(fock, "fock_1h_{:d}".format(self.buff_ix))

            if self.buff_ix == self.splits[-1]:
                self.output.increment()
        self.accumulated = False
        self.zero()

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0

    def setup_output(self, filename):
        est_name = "back_propagated"
        if self.eval_energy:
            with h5py.File(filename, "a") as fh5:
                fh5[est_name + "/headers"] = numpy.array(self.header).astype("S")
        self.output = H5EstimatorHelper(filename, est_name)
