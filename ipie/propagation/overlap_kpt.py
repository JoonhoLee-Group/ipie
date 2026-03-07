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

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


def calc_overlap_single_det_kpt(walkers: "UHFWalkers", trial: "KptSingleDet"):
    """Caculate overlap with single det k point trial wavefunction.

    Parameters
    ----------
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ot : float / complex
        Overlap.
    """
    nwalkers = walkers.nwalkers
    nup = trial.nalpha
    ndown = trial.nbeta
    nk = trial.nk
    nbsf = trial.nbasis

    phia = walkers.phia.reshape(nwalkers, nk, nbsf, nk, nup)
    if ndown > 0 and not walkers.rhf:
        phib = walkers.phib.reshape(nwalkers, nk, nbsf, nk, ndown)

    ovlpa = xp.einsum("wlpki, lpj->wkilj", phia, trial.psi0a.conj(), optimize=True)
    if trial.noccas is not None:
        mask = xp.arange(nup)[None, :] >= xp.array(trial.noccas)[:, None]
        ik_idx, diag_idx = xp.nonzero(mask)
        ovlpa[:, ik_idx, diag_idx, ik_idx, diag_idx] = 1.0

    ovlpa_reshape = ovlpa.reshape((nwalkers, nk * nup, nk * nup))
    sign_a, log_ovlp_a = xp.linalg.slogdet(ovlpa_reshape)

    if ndown > 0 and not walkers.rhf:
        ovlpb = xp.einsum("wlpki, lpj->wkilj", phib, trial.psi0b.conj(), optimize=True)
        if trial.noccbs is not None:
            mask = xp.arange(ndown)[None, :] >= xp.array(trial.noccbs)[:, None]
            ik_idx, diag_idx = xp.nonzero(mask)
            ovlpb[:, ik_idx, diag_idx, ik_idx, diag_idx] = 1.0
        ovlpb_reshape = ovlpb.reshape((nwalkers, nk * ndown, nk * ndown))
        sign_b, log_ovlp_b = xp.linalg.slogdet(ovlpb_reshape)
        ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
        sgnot = sign_a * sign_b
        logot = log_ovlp_a + log_ovlp_b - walkers.log_shift
    elif ndown > 0 and walkers.rhf:
        ot = sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walkers.log_shift)
        sgnot = sign_a * sign_a
        logot = log_ovlp_a + log_ovlp_a - walkers.log_shift
    elif ndown == 0:
        ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)
        sgnot = sign_a
        logot = log_ovlp_a - walkers.log_shift

    synchronize()

    return ot, sgnot, logot
