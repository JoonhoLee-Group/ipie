import json

import h5py
import numpy
import pandas as pd

from ipie.utils.misc import get_from_dict
def set_info(frame, md):
    ncols = len(frame.columns)
    system = md.get("system")
    hamiltonian = md.get("hamiltonian")
    trial = md.get("trial")
    qmc = md.get("params")
    fp = get_from_dict(md, ["propagators", "free_projection"])
    bp = get_from_dict(md, ["estimates", "estimates", "back_prop"])

    beta = qmc.get("beta")
    br = qmc.get("beta_scaled")

    ints = system.get("integral_file")
    chol = system.get("threshold")

    frame["nup"] = system.get("nup")
    frame["ndown"] = system.get("ndown")
    frame["mu"] = qmc.get("mu")
    frame["beta"] = qmc.get("beta")
    frame["dt"] = qmc.get("timestep")
    frame["ntot_walkers"] = qmc.get("total_num_walkers", 0)
    frame["nbasis"] = hamiltonian.get("nbasis", 0)

    if trial is not None:
        frame["mu_T"] = trial.get("mu")
        frame["Nav_T"] = trial.get("nav")

    if fp is not None:
        frame["free_projection"] = fp

    if bp is not None:
        frame["tau_bp"] = bp["tau_bp"]

    if br is not None:
        frame["beta_red"] = br

    if ints is not None:
        frame["integrals"] = ints

    if chol is not None:
        frame["cholesky_treshold"] = chol

    return list(frame.columns[ncols:])

