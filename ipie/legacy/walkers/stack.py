import numpy
import scipy.linalg

from ipie.utils.misc import get_numeric_names


class FieldConfig(object):
    """Object for managing stored auxilliary field.

    Parameters
    ----------
    nfields : int
        Number of fields to store for each back propagation step.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    dtype : type
        Field configuration type.
    """

    def __init__(self, nfields, nprop_tot, nbp, dtype):
        self.configs = numpy.zeros(shape=(nprop_tot, nfields), dtype=dtype)
        self.cos_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=float)
        self.weight_fac = numpy.zeros(shape=(nprop_tot, 1), dtype=complex)
        self.tot_wfac = 1.0 + 0j
        self.step = 0
        # need to account for first iteration and how we iterate
        self.block = -1
        self.ib = 0
        self.nfields = nfields
        self.nbp = nbp
        self.nprop_tot = nprop_tot
        self.nblock = nprop_tot // nbp
        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

    def push(self, config):
        """Add field configuration to buffer.

        Parameters
        ----------
        config : int
            Auxilliary field configuration.
        """
        self.configs[self.step, self.ib] = config
        self.ib = (self.ib + 1) % self.nfields
        # Completed field configuration for this walker?
        if self.ib == 0:
            self.step = self.step + 1
            # Completed this block of back propagation steps?
            if self.step % self.nbp == 0:
                self.block = (self.block + 1) % self.nblock

    def update(self, config, wfac):
        """Add full field configuration for walker to buffer.

        Parameters
        ----------
        config : :class:`numpy.ndarray`
            Auxilliary field configuration.
        cfac : float
            Cosine factor if using phaseless approximation.
        wfac : complex
            Weight factor to restore full walker weight following phaseless
            approximation.
        """
        self.configs[self.step] = config
        self.weight_fac[self.step] = wfac[0]
        # cosine factor is real..
        self.cos_fac[self.step] = wfac[1].real
        if abs(wfac[1]) > 1e-16:
            self.tot_wfac *= wfac[0] / wfac[1]
        else:
            self.tot_wfac = 0
        # Completed field configuration for this walker?
        self.step = self.step + 1
        # Completed this block of back propagation steps?
        if self.step % self.nbp == 0:
            self.block = (self.block + 1) % self.nblock

    def get_block(self):
        """Return a view to current block for back propagation."""
        end = self.step
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])

    def get_superblock(self):
        """Return a view to current super block for ITCF."""
        end = self.nprop_tot - self.nbp
        return (self.configs[:end], self.cos_fac[:end], self.weight_fac[:end])

    def get_buffer(self):
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, (numpy.ndarray)):
                buff[s : s + data.size] = data.ravel()
                s += data.size
            else:
                buff[s : s + 1] = data
                s += 1
        return buff

    def set_buffer(self, buff):
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, numpy.ndarray):
                self.__dict__[d] = buff[s : s + data.size].reshape(data.shape).copy()
                dsize = data.size
            else:
                if isinstance(self.__dict__[d], int):
                    self.__dict__[d] = int(buff[s].real)
                elif isinstance(self.__dict__[d], float):
                    self.__dict__[d] = buff[s].real
                else:
                    self.__dict__[d] = buff[s]
                dsize = 1
            s += dsize

    def get_wfac(self):
        cfac = numpy.prod(self.cos_fac[: self.step])
        wfac = numpy.prod(self.weight_fac[: self.step])
        return cfac, wfac

    def reset(self):
        if self.step % self.nprop_tot == 0:
            self.step = 0
        self.tot_wfac = 1.0 + 0j


class PropagatorStack:
    def __init__(
        self,
        stack_size,
        ntime_slices,
        nbasis,
        dtype,
        BT=None,
        BTinv=None,
        diagonal=False,
        averaging=False,
        lowrank=True,
        thresh=1e-6,
    ):

        self.time_slice = 0
        self.stack_size = stack_size
        self.ntime_slices = ntime_slices
        self.nbins = ntime_slices // self.stack_size
        self.diagonal_trial = diagonal
        self.averaging = averaging
        self.thresh = thresh

        self.lowrank = lowrank
        self.ovlp = numpy.asarray([1.0, 1.0])

        if self.lowrank:
            assert diagonal

        self.reortho = 1

        if self.nbins * self.stack_size < self.ntime_slices:
            print("stack_size must divide the total path length")
            assert self.nbins * self.stack_size == self.ntime_slices

        self.nbasis = nbasis
        self.dtype = dtype
        self.BT = BT
        self.BTinv = BTinv
        self.counter = 0
        self.block = 0

        self.stack = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis), dtype=dtype)
        self.left = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis), dtype=dtype)
        self.right = numpy.zeros(shape=(self.nbins, 2, nbasis, nbasis), dtype=dtype)

        self.G = numpy.asarray(
            [numpy.eye(self.nbasis, dtype=dtype), numpy.eye(self.nbasis, dtype=dtype)]
        )

        if self.lowrank:
            self.update_new = self.update_low_rank
        else:
            self.update_new = self.update_full_rank

        # Global block matrix
        if self.lowrank:
            self.Ql = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.Dl = numpy.zeros(shape=(2, nbasis), dtype=dtype)
            self.Tl = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)

            self.Qr = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.Dr = numpy.zeros(shape=(2, nbasis), dtype=dtype)
            self.Tr = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)

            self.CT = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.theta = numpy.zeros(shape=(2, nbasis, nbasis), dtype=dtype)
            self.mT = nbasis

        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)
        # set all entries to be the identity matrix
        self.reset()

    def get(self, ix):
        return self.stack[ix]

    def get_buffer(self):
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, (numpy.ndarray)):
                buff[s : s + data.size] = data.ravel()
                s += data.size
            else:
                buff[s : s + 1] = data
                s += 1
        return buff

    def set_buffer(self, buff):
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, numpy.ndarray):
                self.__dict__[d] = buff[s : s + data.size].reshape(data.shape).copy()
                dsize = data.size
            else:
                if isinstance(self.__dict__[d], int):
                    self.__dict__[d] = int(buff[s].real)
                elif isinstance(self.__dict__[d], float):
                    self.__dict__[d] = float(buff[s].real)
                else:
                    self.__dict__[d] = buff[s]
                dsize = 1
            s += dsize

    def set_all(self, BT):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.diagonal_trial:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size  # bin index
                # Commenting out these two. It is only useful for Hubbard
                self.left[ix, 0] = numpy.diag(
                    numpy.multiply(BT[0].diagonal(), self.left[ix, 0].diagonal())
                )
                self.left[ix, 1] = numpy.diag(
                    numpy.multiply(BT[1].diagonal(), self.left[ix, 1].diagonal())
                )
                self.stack[ix, 0] = self.left[ix, 0].copy()
                self.stack[ix, 1] = self.left[ix, 1].copy()
        else:
            for i in range(0, self.ntime_slices):
                ix = i // self.stack_size  # bin index
                self.left[ix, 0] = numpy.dot(BT[0], self.left[ix, 0])
                self.left[ix, 1] = numpy.dot(BT[1], self.left[ix, 1])
                self.stack[ix, 0] = self.left[ix, 0].copy()
                self.stack[ix, 1] = self.left[ix, 1].copy()

        if self.lowrank:
            self.initialize_left()
            for s in [0, 1]:
                self.Qr[s] = numpy.identity(self.nbasis, dtype=self.dtype)
                self.Dr[s] = numpy.ones(self.nbasis, dtype=self.dtype)
                self.Tr[s] = numpy.identity(self.nbasis, dtype=self.dtype)

    def reset(self):
        self.time_slice = 0
        self.block = 0
        for i in range(0, self.nbins):
            self.stack[i, 0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.stack[i, 1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i, 0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.right[i, 1] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i, 0] = numpy.identity(self.nbasis, dtype=self.dtype)
            self.left[i, 1] = numpy.identity(self.nbasis, dtype=self.dtype)

        if self.lowrank:
            for s in [0, 1]:
                self.Qr[s] = numpy.identity(self.nbasis, dtype=self.dtype)
                self.Dr[s] = numpy.ones(self.nbasis, dtype=self.dtype)
                self.Tr[s] = numpy.identity(self.nbasis, dtype=self.dtype)

    # Form BT product for i = 1, ..., nslices - 1 (i.e., skip i = 0)
    # \TODO add non-diagonal version of this
    def initialize_left(self):
        assert self.diagonal_trial
        for spin in [0, 1]:
            # We will assume that B matrices are all diagonal for left....
            # B = self.stack[1]
            B = self.stack[0]
            self.Dl[spin] = B[spin].diagonal()
            self.Ql[spin] = numpy.identity(B[spin].shape[0])
            self.Tl[spin] = numpy.identity(B[spin].shape[0])

            # for ix in range(2, self.nbins):
            for ix in range(1, self.nbins):
                B = self.stack[ix]
                C2 = numpy.einsum("ii,i->i", B[spin], self.Dl[spin])
                self.Dl[spin] = C2

    def update(self, B):
        if self.counter == 0:
            self.stack[self.block, 0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.stack[self.block, 1] = numpy.identity(B.shape[-1], dtype=B.dtype)
        self.stack[self.block, 0] = B[0].dot(self.stack[self.block, 0])
        self.stack[self.block, 1] = B[1].dot(self.stack[self.block, 1])
        self.time_slice = self.time_slice + 1
        self.block = self.time_slice // self.stack_size
        self.counter = (self.counter + 1) % self.stack_size

    def update_full_rank(self, B):
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        if self.counter == 0:
            self.right[self.block, 0] = numpy.identity(B.shape[-1], dtype=B.dtype)
            self.right[self.block, 1] = numpy.identity(B.shape[-1], dtype=B.dtype)

        if self.diagonal_trial:
            self.left[self.block, 0] = numpy.diag(
                numpy.multiply(
                    self.left[self.block, 0].diagonal(), self.BTinv[0].diagonal()
                )
            )
            self.left[self.block, 1] = numpy.diag(
                numpy.multiply(
                    self.left[self.block, 1].diagonal(), self.BTinv[1].diagonal()
                )
            )
        else:
            self.left[self.block, 0] = self.left[self.block, 0].dot(self.BTinv[0])
            self.left[self.block, 1] = self.left[self.block, 1].dot(self.BTinv[1])

        self.right[self.block, 0] = B[0].dot(self.right[self.block, 0])
        self.right[self.block, 1] = B[1].dot(self.right[self.block, 1])

        if self.diagonal_trial:
            self.stack[self.block, 0] = numpy.einsum(
                "ii,ij->ij", self.left[self.block, 0], self.right[self.block, 0]
            )
            self.stack[self.block, 1] = numpy.einsum(
                "ii,ij->ij", self.left[self.block, 1], self.right[self.block, 1]
            )
        else:
            self.stack[self.block, 0] = self.left[self.block, 0].dot(
                self.right[self.block, 0]
            )
            self.stack[self.block, 1] = self.left[self.block, 1].dot(
                self.right[self.block, 1]
            )

        self.time_slice = self.time_slice + 1  # Count the time slice
        self.block = (
            self.time_slice // self.stack_size
        )  # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size  # Counting within a stack

    def update_low_rank(self, B):
        assert not self.averaging
        # Diagonal = True assumes BT is diagonal and left is also diagonal
        assert self.diagonal_trial

        if self.counter == 0:
            for s in [0, 1]:
                self.Tl[s] = self.left[self.block, s]

        mR = B.shape[-1]  # initial mR
        mL = B.shape[-1]  # initial mR
        mT = B.shape[-1]  # initial mR
        next_block = (
            self.time_slice + 1
        ) // self.stack_size  # move to the next block if necessary
        # print("next_block", next_block)
        # print("self.block", self.block)
        if next_block > self.block:  # Do QR and update here?
            for s in [0, 1]:
                mR = len(self.Dr[s][numpy.abs(self.Dr[s]) > self.thresh])
                self.Dl[s] = numpy.einsum("i,ii->i", self.Dl[s], self.BTinv[s])
                mL = len(self.Dl[s][numpy.abs(self.Dl[s]) > self.thresh])

                self.Qr[s][:, :mR] = B[s].dot(self.Qr[s][:, :mR])  # N x mR
                self.Qr[s][:, mR:] = 0.0

                Ccr = numpy.einsum(
                    "ij,j->ij", self.Qr[s][:, :mR], self.Dr[s][:mR]
                )  # N x mR
                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(
                    Ccr, pivoting=True, check_finite=False
                )
                Dlcr = Rlcr[:mR, :mR].diagonal()  # mR

                self.Dr[s][:mR] = Dlcr
                self.Dr[s][mR:] = 0.0
                self.Qr[s] = Qlcr

                Dinv = 1.0 / Dlcr  # mR
                tmp = numpy.einsum(
                    "i,ij->ij", Dinv[:mR], Rlcr[:mR, :mR]
                )  # mR, mR x mR -> mR x mR
                tmp[:, Plcr] = tmp[:, range(mR)]
                Tlcr = numpy.dot(tmp, self.Tr[s][:mR, :])  # mR x N

                self.Tr[s][:mR, :] = Tlcr

                # assume left stack is all diagonal (i.e., QDT = diagonal -> Q and T are identity)
                Clcr = numpy.einsum(
                    "i,ij->ij",
                    self.Dl[s][:mL],
                    numpy.einsum("ij,j->ij", Qlcr[:mL, :mR], Dlcr[:mR]),
                )  # mL x mR

                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(
                    Clcr, pivoting=True, check_finite=False
                )  # mL x mL, min(mL,mR) x min(mL,mR), mR x mR
                Dlcr = Rlcr.diagonal()[: min(mL, mR)]
                Dinv = 1.0 / Dlcr

                mT = len(Dlcr[numpy.abs(Dlcr) > self.thresh])

                assert mT <= mL and mT <= mR

                tmp = numpy.einsum("i,ij->ij", Dinv[:mT], Rlcr[:mT, :])
                tmp[:, Plcr] = tmp[:, range(mR)]  # mT x mR
                Tlcr = numpy.dot(tmp, Tlcr)  # mT x N

                Db = numpy.zeros(mT, B[s].dtype)
                Ds = numpy.zeros(mT, B[s].dtype)
                for i in range(mT):
                    absDlcr = abs(Dlcr[i])
                    if absDlcr > 1.0:
                        Db[i] = 1.0 / absDlcr
                        Ds[i] = numpy.sign(Dlcr[i])
                    else:
                        Db[i] = 1.0
                        Ds[i] = Dlcr[i]
                Dbinv = 1.0 / Db

                TQ = Tlcr[:, :mL].dot(Qlcr[:mL, :mT])  # mT x mT
                TQinv = scipy.linalg.inv(TQ, check_finite=False)
                tmp = numpy.einsum("ij,j->ij", TQinv, Db) + numpy.diag(Ds)  # mT x mT

                M = numpy.einsum("ij,j->ij", tmp, Dbinv).dot(TQ)
                # self.ovlp[s] = 1.0 / scipy.linalg.det(M, check_finite=False)
                self.ovlp[s] = scipy.linalg.det(M, check_finite=False)

                tmp = scipy.linalg.inv(tmp, check_finite=False)
                A = numpy.einsum("i,ij->ij", Db, tmp.dot(TQinv))  # mT x mT
                Qlcr_pad = numpy.zeros((self.nbasis, self.nbasis), dtype=B[s].dtype)
                Qlcr_pad[:mL, :mT] = Qlcr[:, :mT]

                # self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - Qlcr_pad[:,:mT].dot(numpy.diag(Dlcr[:mT])).dot(A).dot(Tlcr)

                self.CT[s][:, :] = 0.0
                self.CT[s][:, :mT] = (A.dot(Tlcr)).T.conj()
                self.theta[s][:, :] = 0.0
                self.theta[s][:mT, :] = Qlcr_pad[:, :mT].dot(numpy.diag(Dlcr[:mT])).T
                # self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - self.CT[s][:,:mT].dot(self.theta[s][:mT,:])
                self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - self.theta[s][
                    :mT, :
                ].T.dot(self.CT[s][:, :mT].T.conj())
                # self.CT[s][:,:mT] = self.CT[s][:,:mT].conj()

                # print("# mL, mR, mT = {}, {}, {}".format(mL, mR, mT))
        else:  # don't do QR and just update
            for s in [0, 1]:
                mR = len(self.Dr[s][numpy.abs(self.Dr[s]) > self.thresh])

                self.Dl[s] = numpy.einsum("i,ii->i", self.Dl[s], self.BTinv[s])
                mL = len(self.Dl[s][numpy.abs(self.Dl[s]) > self.thresh])

                self.Qr[s][:, :mR] = B[s].dot(self.Qr[s][:, :mR])  # N x mR
                self.Qr[s][:, mR:] = 0.0

                Ccr = numpy.einsum(
                    "ij,j->ij", self.Qr[s][:, :mR], self.Dr[s][:mR]
                )  # N x mR
                Clcr = numpy.einsum(
                    "i,ij->ij", self.Dl[s][:mL], Ccr[:mL, :mR]
                )  # mL x mR

                (Qlcr, Rlcr, Plcr) = scipy.linalg.qr(
                    Clcr, pivoting=True, check_finite=False
                )  # mL x mL, min(mL,mR) x min(mL,mR), mR x mR
                Dlcr = Rlcr.diagonal()[: min(mL, mR)]
                Dinv = 1.0 / Dlcr

                mT = len(Dlcr[numpy.abs(Dlcr) > self.thresh])

                assert mT <= mL and mT <= mR

                tmp = numpy.einsum("i,ij->ij", Dinv[:mT], Rlcr[:mT, :])
                tmp[:, Plcr] = tmp[:, range(mR)]  # mT x mR
                Tlcr = numpy.dot(tmp, self.Tr[s][:mR, :])  # mT x N

                Db = numpy.zeros(mT, B[s].dtype)
                Ds = numpy.zeros(mT, B[s].dtype)
                for i in range(mT):
                    absDlcr = abs(Dlcr[i])
                    if absDlcr > 1.0:
                        Db[i] = 1.0 / absDlcr
                        Ds[i] = numpy.sign(Dlcr[i])
                    else:
                        Db[i] = 1.0
                        Ds[i] = Dlcr[i]
                Dbinv = 1.0 / Db

                TQ = Tlcr[:, :mL].dot(Qlcr[:mL, :mT])  # mT x mT
                TQinv = scipy.linalg.inv(TQ, check_finite=False)
                tmp = numpy.einsum("ij,j->ij", TQinv, Db) + numpy.diag(Ds)  # mT x mT

                M = numpy.einsum("ij,j->ij", tmp, Dbinv).dot(TQ)
                # self.ovlp[s] = 1.0 / scipy.linalg.det(M, check_finite=False)
                self.ovlp[s] = scipy.linalg.det(M, check_finite=False)

                tmp = scipy.linalg.inv(tmp, check_finite=False)
                A = numpy.einsum("i,ij->ij", Db, tmp.dot(TQinv))  # mT x mT
                Qlcr_pad = numpy.zeros((self.nbasis, self.nbasis), dtype=B[s].dtype)
                Qlcr_pad[:mL, :mT] = Qlcr[:, :mT]

                # self.CT[s][:,:] = 0.0
                # self.CT[s][:,:mT] = Qlcr_pad[:,:mT].dot(numpy.diag(Dlcr[:mT]))
                # self.theta[s][:,:] = 0.0
                # self.theta[s][:mT,:] = A.dot(Tlcr)
                # self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - self.CT[s][:,:mT].dot(self.theta[s][:mT,:])
                # self.CT[s][:,:mT] = self.CT[s][:,:mT].conj()
                self.CT[s][:, :] = 0.0
                self.CT[s][:, :mT] = (A.dot(Tlcr)).T.conj()
                self.theta[s][:, :] = 0.0
                self.theta[s][:mT, :] = Qlcr_pad[:, :mT].dot(numpy.diag(Dlcr[:mT])).T
                # self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - self.CT[s][:,:mT].dot(self.theta[s][:mT,:])
                self.G[s] = numpy.eye(self.nbasis, dtype=B[s].dtype) - self.theta[s][
                    :mT, :
                ].T.dot(self.CT[s][:, :mT].T.conj())

            # self.CT = numpy.zeros(shape=(2, nbasis, nbasis),dtype=dtype)
            # self.theta = numpy.zeros(shape=(2, nbasis, nbasis),dtype=dtype)
        # print("# mL, mR, mT = {}, {}, {}".format(mL, mR, mT))

        # print("ovlp = {}".format(self.ovlp))
        self.mT = mT
        self.time_slice = self.time_slice + 1  # Count the time slice
        self.block = (
            self.time_slice // self.stack_size
        )  # move to the next block if necessary
        self.counter = (self.counter + 1) % self.stack_size  # Counting within a stack
