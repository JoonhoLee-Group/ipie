import plum
from ipie.utils.backend import arraylib as xp
from ipie.utils.misc import is_cupy

from ipie.systems.generic import Generic
from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC
from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from ipie.addons.ithc.walkers.uhf_walkers import UHFWalkers


def greens_function_ithc(psi0_extended, GHalf_ori, isometry):
    """Calculates the single particle greens function in an extended basis set"""

    # nori is unused, so we underscore it
    _, nextend = isometry.shape
    nwalkers, _, _ = GHalf_ori.shape  # nelecs and nori unused here

    if is_cupy(psi0_extended):
        GHalf_extended = xp.einsum("wip,pa->wia", GHalf_ori, isometry)
        G_extended = xp.einsum("ai,wib->wab", psi0_extended.conj(), GHalf_extended)

    else:
        G_extended = xp.zeros(shape=(nwalkers, nextend, nextend), dtype=xp.complex128)
        for iw in range(nwalkers):
            GHalf_extended = GHalf_ori[iw] @ isometry
            G_extended[iw] = psi0_extended.conj() @ GHalf_extended

    return G_extended


def compute_pe_batched(trial, walkers, isometry, W, batch_size=32):
    """Pre computes the two-body energy"""
    nwalkers = walkers.Ghalfa.shape[0]
    e2 = xp.empty(nwalkers, dtype=walkers.Ghalfa.dtype)

    for start in range(0, nwalkers, batch_size):
        end = min(start + batch_size, nwalkers)

        Ga = greens_function_ithc(trial._psi0a_transformed, walkers.Ghalfa[start:end], isometry)
        Gb = greens_function_ithc(trial._psi0b_transformed, walkers.Ghalfb[start:end], isometry)

        na = xp.einsum("wii->wi", Ga, optimize=True)
        nb = xp.einsum("wii->wi", Gb, optimize=True)

        exchange_aa = xp.einsum("ij,wij,wji->w", W, Ga, Ga, optimize=True)
        exchange_bb = xp.einsum("ij,wij,wji->w", W, Gb, Gb, optimize=True)

        e2_aa = 0.5 * (xp.einsum("ij,wi,wj->w", W, na, na) - exchange_aa)
        e2_bb = 0.5 * (xp.einsum("ij,wi,wj->w", W, nb, nb) - exchange_bb)
        e2_ab = xp.einsum("ij,wi,wj->w", W, na, nb)

        e2[start:end] = e2_aa + e2_bb + e2_ab

        del Ga, Gb, na, nb, exchange_aa, exchange_bb, e2_aa, e2_bb, e2_ab

    return e2


@plum.dispatch
def local_energy_single_det_uhf_ithc(
    system: Generic, hamiltonian: GenericITHC, walkers: UHFWalkers, trial: SingleDet
):
    """Computes the local energy of a single SD for ithc"""
    isometry = hamiltonian.isometry
    W = hamiltonian.W

    # nbasis and M (nfields) were flagged as unused variables.
    # We only need nfields for the batch_size logic.
    _, nfields = xp.shape(isometry)
    nwalkers = walkers.phia.shape[0]

    # compute the kinetic energy in the original basis
    trial.calc_greens_function(walkers)

    Ghalfa_batch = walkers.Ghalfa.reshape((nwalkers, -1))
    Ghalfb_batch = walkers.Ghalfb.reshape((nwalkers, -1))

    e1 = Ghalfa_batch.dot(trial._rH1a.ravel())
    e1 += Ghalfb_batch.dot(trial._rH1b.ravel())

    if nfields < 64:
        batch_size = min(nwalkers, 1024)
    else:
        # Aim for ~2GB per batch limit
        batch_size = max(32, int(2 * 1024**3 // (16 * nfields**2)))

    if is_cupy(trial._psi0a_transformed):
        e2 = compute_pe_batched(trial, walkers, isometry, W, batch_size=batch_size)
    else:
        e2 = compute_pe_batched(trial, walkers, isometry, W, batch_size=1)

    etot = e1 + e2 + hamiltonian.ecore

    return xp.stack((etot, e1, e2), axis=1)
