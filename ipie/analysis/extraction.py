import json

import h5py
import numpy
import pandas as pd

from ipie.utils.misc import get_from_dict


def extract_data_sets(files, group, estimator, raw=False):
    data = []
    for f in files:
        data.append(extract_data(f, group, estimator, raw))
    return pd.concat(data)


def extract_data(filename, group, estimator, raw=False):
    fp = get_param(filename, ["propagators", "free_projection"])
    with h5py.File(filename, "r") as fh5:
        dsets = list(fh5[group][estimator].keys())
        data = numpy.array([fh5[group][estimator][d][:] for d in dsets])
        if "rdm" in estimator or raw:
            return data
        else:
            header = fh5[group]["headers"][:]
            header = numpy.array([h.decode("utf-8") for h in header])
            df = pd.DataFrame(data)
            df.columns = header
            if not fp:
                df = df.apply(numpy.real)
            return df


def extract_mixed_estimates(filename, skip=0):
    return extract_data(filename, "basic", "energies")[skip:]


def extract_bp_estimates(filename, skip=0):
    return extract_data(filename, "back_propagated", "energies")[skip:]


def extract_rdm(filename, est_type="back_propagated", rdm_type="one_rdm", ix=None):
    rdmtot = []
    if ix is None:
        splits = get_param(
            filename, ["estimators", "estimators", "back_prop", "splits"]
        )
        ix = splits[0][-1]
    if est_type == "back_propagated":
        denom = extract_data(filename, est_type, "denominator_{}".format(ix), raw=True)
        one_rdm = extract_data(
            filename, est_type, rdm_type + "_{}".format(ix), raw=True
        )
    else:
        one_rdm = extract_data(filename, est_type, rdm_type, raw=True)
        denom = None
    fp = get_param(filename, ["propagators", "free_projection"])
    if fp:
        print("# Warning analysis of FP RDM not implemented.")
        return (one_rdm, denom)
    else:
        if denom is None:
            return one_rdm
        if len(one_rdm.shape) == 4:
            return one_rdm / denom[:, None, None]
        elif len(one_rdm.shape) == 5:
            return one_rdm / denom[:, None, None, None]
        elif len(one_rdm.shape) == 3:
            return one_rdm / denom[:, None]
        else:
            return one_rdm / denom


def set_info(frame, md):
    system = md.get("system")
    qmc = md.get("qmc")
    propg = md.get("propagators")
    trial = md.get("trial")
    ncols = len(frame.columns)
    frame["dt"] = qmc.get("dt")
    nwalkers = qmc.get("ntot_walkers")
    if nwalkers is not None:
        frame["nwalkers"] = nwalkers
    fp = get_from_dict(md, ["propagators", "free_projection"])
    if fp is not None:
        frame["free_projection"] = fp
    beta = qmc.get("beta")
    bp = get_from_dict(md, ["estimates", "estimates", "back_prop"])
    frame["nbasis"] = system.get("nbasis", 0)
    if bp is not None:
        frame["tau_bp"] = bp["tau_bp"]
    if beta is not None:
        frame["beta"] = beta
        br = qmc.get("beta_scaled")
        if br is not None:
            frame["beta_red"] = br
        mu = system.get("mu")
        if mu is not None:
            frame["mu"] = system.get("mu")
        if trial is not None:
            frame["mu_T"] = trial.get("mu")
            frame["Nav_T"] = trial.get("nav")
    else:
        frame["E_T"] = trial.get("energy")
    if system["name"] == "UEG":
        frame["rs"] = system.get("rs")
        frame["ecut"] = system.get("ecut")
        frame["nup"] = system.get("nup")
        frame["ndown"] = system["ndown"]
    elif system["name"] == "Hubbard":
        frame["U"] = system.get("U")
        frame["nx"] = system.get("nx")
        frame["ny"] = system.get("ny")
        frame["nup"] = system.get("nup")
        frame["ndown"] = system.get("ndown")
    elif system["name"] == "Generic":
        ints = system.get("integral_file")
        if ints is not None:
            frame["integrals"] = ints
        chol = system.get("threshold")
        if chol is not None:
            frame["cholesky_treshold"] = chol
        frame["nup"] = system.get("nup")
        frame["ndown"] = system.get("ndown")
        frame["nbasis"] = system.get("nbasis", 0)
    return list(frame.columns[ncols:])


def get_metadata(filename):
    try:
        with h5py.File(filename, "r") as fh5:
            metadata = json.loads(fh5["metadata"][()])
    except:
        print("# problem with file = {}".format(filename))
    return metadata


def get_param(filename, param):
    md = get_metadata(filename)
    return get_from_dict(md, param)


def get_sys_param(filename, param):
    return get_param(filename, ["system", param])


def extract_test_data_hdf5(filename, skip=10):
    """For use with testcode"""
    data = extract_mixed_estimates(filename)
    # use list so can json serialise easily.
    data = data.drop(["Iteration", "Time"], axis=1)[::skip].to_dict(orient="list")
    data["sys_info"] = get_metadata(filename)["sys_info"]
    try:
        mrdm = extract_rdm(filename, est_type="mixed", rdm_type="one_rdm")
    except (KeyError, TypeError, AttributeError):
        mrdm = None
    try:
        brdm = extract_rdm(filename, est_type="back_propagated", rdm_type="one_rdm")
    except (KeyError, TypeError, AttributeError):
        brdm = None
    if mrdm is not None:
        mrdm = mrdm[::4].ravel()
        # Don't compare small numbers
        re = numpy.real(mrdm)
        im = numpy.imag(mrdm)
        re[numpy.abs(re) < 1e-12] = 0.0
        im[numpy.abs(im) < 1e-12] = 0.0
        data["Gmixed_re"] = list(mrdm)
        data["Gmixed_im"] = list(mrdm)
    if brdm is not None:
        brdm = brdm[::4].flatten().copy()
        re = numpy.real(brdm)
        im = numpy.imag(brdm)
        re[numpy.abs(re) < 1e-12] = 0.0
        im[numpy.abs(im) < 1e-12] = 0.0
        data["Gbp_re"] = list(re)
        data["Gbp_im"] = list(im)
    # if itcf is not None:
    # itcf = itcf[abs(itcf) > 1e-10].flatten()
    # data = pd.DataFrame(itcf)
    return data
