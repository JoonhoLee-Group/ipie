import numpy

from ipie.legacy.estimators.thermal import one_rdm_stable, particle_number
from ipie.utils.io import format_fixed_width_floats, format_fixed_width_strings


def find_chemical_potential(
    alt_convention, rho, beta, num_bins, target, deps=1e-6, max_it=1000, verbose=False
):
    # Todo: some sort of generic starting point independent of
    # system/temperature
    dmu1 = dmu2 = 1
    mu1 = -1
    mu2 = 1
    sign = -1 if alt_convention else 1
    if verbose:
        print("# Finding chemical potential to match <N> = {:13.8e}".format(target))
    while numpy.sign(dmu1) * numpy.sign(dmu2) > 0:
        rho1 = compute_rho(rho, mu1, beta, sign=sign)
        dmat = one_rdm_stable(rho1, num_bins)
        dmu1 = delta_nav(dmat, target)
        rho2 = compute_rho(rho, mu2, beta, sign=sign)
        dmat = one_rdm_stable(rho2, num_bins)
        dmu2 = delta_nav(dmat, target)
        if numpy.sign(dmu1) * numpy.sign(dmu2) < 0:
            if verbose:
                print("# Chemical potential lies within range of [%f,%f]" % (mu1, mu2))
                print("# delta_mu1 = %f, delta_mu2 = %f" % (dmu1.real, dmu2.real))
            break
        else:
            mu1 -= 2
            mu2 += 2
            if verbose:
                print("# Increasing chemical potential search to [%f,%f]" % (mu1, mu2))
    found_mu = False
    if verbose:
        print("# " + format_fixed_width_strings(["iteration", "mu", "Dmu", "<N>"]))
    for i in range(0, max_it):
        mu = 0.5 * (mu1 + mu2)
        rho_mu = compute_rho(rho, mu, beta, sign=sign)
        dmat = one_rdm_stable(rho_mu, num_bins)
        dmu = delta_nav(dmat, target).real
        if verbose:
            out = [i, mu, dmu, particle_number(dmat).real]
            print("# " + format_fixed_width_floats(out))
        if abs(dmu) < deps:
            found_mu = True
            break
        else:
            if dmu * dmu1 > 0:
                mu1 = mu
            elif dmu * dmu2 > 0:
                mu2 = mu
    if found_mu:
        return mu
    else:
        print("# Error chemical potential not found")
        return None


def delta_nav(dm, nav):
    return particle_number(dm) - nav


def compute_rho(rho, mu, beta, sign=1):
    return numpy.einsum(
        "ijk,k->ijk", rho, numpy.exp(sign * beta * mu * numpy.ones(rho.shape[-1]))
    )
