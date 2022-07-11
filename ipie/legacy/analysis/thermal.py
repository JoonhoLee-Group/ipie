import numpy
import pandas as pd
import scipy.optimize
import scipy.stats

from ipie.analysis.blocking import average_ratio
from ipie.analysis.extraction import extract_data, get_metadata, set_info


def analyse_energy(files):
    sims = []
    files = sorted(files)
    for f in files:
        data = extract_data(f, "basic", "energies")
        md = get_metadata(f)
        keys = set_info(data, md)
        sims.append(data[1:])
    full = pd.concat(sims).groupby(keys, sort=False)
    analysed = []
    for (i, g) in full:
        if g["free_projection"].values[0]:
            cols = ["ENumer", "Nav"]
            obs = ["ETotal", "Nav"]
            averaged = pd.DataFrame(index=[0])
            for (c, o) in zip(cols, obs):
                (value, error) = average_ratio(g[c].values, g["EDenom"].values)
                averaged[o] = [value]
                averaged[o + "_error"] = [error]
            for (k, v) in zip(full.keys, i):
                averaged[k] = v
            analysed.append(averaged)
        else:
            cols = ["ETotal", "E1Body", "E2Body", "Nav"]
            averaged = pd.DataFrame(index=[0])
            for c in cols:
                mean = numpy.real(g[c].values).mean()
                error = scipy.stats.sem(numpy.real(g[c].values), ddof=1)
                averaged[c] = [mean]
                averaged[c + "_error"] = [error]
            for (k, v) in zip(full.keys, i):
                averaged[k] = v
            analysed.append(averaged)
    return pd.concat(analysed).reset_index(drop=True).sort_values(by=keys)


def nav_mu(mu, coeffs):
    return numpy.polyval(coeffs, mu)


def find_chem_pot(data, target, vol, order=3, plot=False):
    print("# System volume: {}.".format(vol))
    print("# Target number of electrons: {}.".format(vol * target))
    nav = data.Nav.values / vol
    nav_error = data.Nav_error.values / vol
    # Half filling special case where error bar is zero.
    zeros = numpy.where(nav_error == 0)[0]
    nav_error[zeros] = 1e-8
    mus = data.mu.values
    delta = nav - target
    s = 0
    e = len(delta)
    rmin = None
    while e - s > order + 1:
        mus_range = mus[s:e]
        delta_range = delta[s:e]
        err_range = nav_error[s:e]
        fit, res, rk, sv, rcond = numpy.polyfit(
            mus_range, delta_range, order, w=1.0 / err_range, full=True
        )
        a = min(mus_range)
        b = max(mus_range)
        try:
            mu, r = scipy.optimize.brentq(nav_mu, a, b, args=fit, full_output=True)
            if rmin is None:
                rmin = res[0]
                mu_min = mu
            elif res[0] < rmin:
                mu_min = mu
            print(
                "# min = {:f} max = {:f} res = {:} mu = " "{:f}".format(a, b, res, mu)
            )
        except ValueError:
            mu = None
            print("Root not found in interval.")
        s += 1
        e -= 1

    if plot:
        import matplotlib.pyplot as pl

        beta = data.beta[0]
        pl.errorbar(
            mus,
            delta,
            yerr=nav_error,
            fmt="o",
            label=r"$\beta = {}$".format(beta),
            color="C0",
        )
        xs = numpy.linspace(a, b, 101)
        ys = nav_mu(xs, fit)
        pl.plot(xs, ys, ":", color="C0")
        if mu is not None and r.converged:
            pl.axvline(mu, linestyle=":", label=r"$\mu^* = {}$".format(mu), color="C3")
        pl.xlabel(r"$\mu$")
        pl.ylabel(r"$n-n_{\mathrm{av}}$")
        pl.legend(numpoints=1)
        pl.show()
    if mu is not None:
        if r.converged:
            return mu_min
        else:
            return None
