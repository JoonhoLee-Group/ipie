import numpy

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, qr, qr_mode, synchronize
from ipie.walkers.base_walkers import BaseWalkers
from ipie.trial_wavefunction.holstein.toyozawa import ToyozawaTrial

class EphWalkers(BaseWalkers):
    """"""
    def __init__(
        self, 
        initial_walker: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        mpi_handler,
        verbose: bool = False
    ):

        self.nup = nup
        self.ndown = ndown
        self.nbasis = nbasis
        self.mpi_handler = mpi_handler


        super().__init__(nwalkers, verbose=verbose)

        self.weight = numpy.ones(self.nwalkers, dtype=numpy.complex128)

        #TODO is there a reason we dont use numpy tile for these?
        self.phia = xp.array(
            [initial_walker[:, : self.nup].copy() for iw in range(self.nwalkers)],
            dtype=xp.complex128,
        )
#        self.phia = numpy.squeeze(self.phia) #NOTE: 1e hack to work with 1e overlaps

        self.phib = xp.array(
            [initial_walker[:, self.nup:self.nup+self.ndown].copy() for iw in range(self.nwalkers)],
            dtype=xp.complex128,
        )
        self.x = xp.array(
            [initial_walker[:, self.nup+self.ndown:].copy() for iw in range(self.nwalkers)],
            dtype=xp.complex128
        )
        self.x = numpy.squeeze(self.x)
        
        self.buff_names += ["phia", "phib", "x"]

        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def build(self, trial):
        """NOTE: total_ovlp is the same as ovlp for coherent state trial, and just
        serves the purpose of not recomputing overlaps for each permutation 
        but passing it to the pop_control and adjsuting it accordingly for the 
        Toyozawa trial."""

        if isinstance(trial, ToyozawaTrial):
            shape = (self.nwalkers, trial.nperms)
        else:
            shape = self.nwalkers

        self.ph_ovlp = numpy.zeros(shape, dtype=numpy.complex128)
        self.el_ovlp = numpy.zeros(shape, dtype=numpy.complex128)
        self.total_ovlp = numpy.zeros(shape, dtype=numpy.complex128)

        self.buff_names += ['total_ovlp'] #, 'el_ovlp', 'total_ovlp'] #not really necessary to bring 'el_ovlp', 'total_ovlp' along if computing overlap after normalization anyways.
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def reortho(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        ndown = self.ndown
        detR = []
        for iw in range(self.nwalkers):
            (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            Rup_diag = xp.diag(Rup)
            signs_up = xp.sign(Rup_diag)
            self.phia[iw] = xp.dot(self.phia[iw], xp.diag(signs_up))

            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = xp.sum(xp.log(xp.abs(Rup_diag)))

            if ndown > 0:
                (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                Rdn_diag = xp.diag(Rdn)
                signs_dn = xp.sign(Rdn_diag)
                self.phib[iw] = xp.dot(self.phib[iw], xp.diag(signs_dn))
                log_det += sum(xp.log(abs(Rdn_diag)))

            detR += [xp.exp(log_det - self.detR_shift[iw])]
            self.log_detR[iw] += xp.log(detR[iw])
            self.detR[iw] = detR[iw]
            
            self.el_ovlp[iw, :] = self.el_ovlp[iw, :] / detR[iw]
            self.total_ovlp[iw, :] = self.total_ovlp[iw, :] / detR[iw]
            self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return detR

    def reortho_batched(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        assert config.get_option("use_gpu")
        (self.phia, Rup) = qr(self.phia, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi", Rup)
        log_det = xp.einsum("wi->w", xp.log(abs(Rup_diag)))

        if self.ndown > 0:
            (self.phib, Rdn) = qr(self.phib, mode=qr_mode)
            Rdn_diag = xp.einsum("wii->wi", Rdn)
            log_det += xp.einsum("wi->w", xp.log(abs(Rdn_diag)))
        self.detR = xp.exp(log_det - self.detR_shift)
        self.ovlp = self.ovlp / self.detR

        synchronize()

        return self.detR


