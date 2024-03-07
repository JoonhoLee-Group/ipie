# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import qr, qr_mode, synchronize
from ipie.walkers.uhf_walkers import UHFWalkers


class UHFWalkersFP(UHFWalkers):
    """UHF style walker specialized for its use with free projection."""

    def orthogonalise(self, free_projection=False):
        """Orthogonalise all walkers.

        Parameters
        ----------
        free_projection : bool
            This flag is not used here.
        """
        detR = self.reortho()
        magn, dtheta = xp.abs(self.detR), xp.angle(self.detR)
        self.weight *= magn
        self.phase *= xp.exp(1j * dtheta)
        return detR

    def reortho_batched(self):
        assert config.get_option("use_gpu")
        (self.phia, Rup) = qr(self.phia, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi", Rup)
        det = xp.prod(Rup_diag, axis=1)

        if self.ndown > 0:
            (self.phib, Rdn) = qr(self.phib, mode=qr_mode)
            Rdn_diag = xp.einsum("wii->wi", Rdn)
            det *= xp.prod(Rdn_diag, axis=1)
        self.detR = det
        self.ovlp = self.ovlp / self.detR
        synchronize()
        return self.detR

    def reortho(self):
        """reorthogonalise walkers for free projection, retaining normalization.

        parameters
        ----------
        """
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        else:
            ndown = self.ndown
            detR = []
            for iw in range(self.nwalkers):
                (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
                det_i = xp.prod(xp.diag(Rup))

                if ndown > 0:
                    (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                    det_i *= xp.prod(xp.diag(Rdn))

                detR += [det_i]
                self.log_detR[iw] += xp.log(detR[iw])
                self.detR[iw] = detR[iw]
                self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return self.detR
