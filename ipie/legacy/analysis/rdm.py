import numpy
import pandas as pd

from ipie.analysis.extraction import extract_rdm, get_param


def analyse_split(A, Ps):
    vals = [numpy.einsum("ij,ij->", A, P[0] + P[1]).real for P in Ps]
    mean = numpy.mean(vals)
    err = numpy.std(vals, ddof=1) / len(vals) ** 0.5
    return mean, err


def analyse_one_body(
    filename, one_body, est_type="back_propagated", rdm_type="one_rdm", skip=1
):
    """Contract one-body operator with QMC rdm and estimate errors."""
    if len(one_body.shape) == 3:
        spin_dep = True
    else:
        spin_dep = False
    splits = get_param(filename, ["estimators", "estimators", "back_prop", "splits"])[0]
    dt = get_param(filename, ["qmc", "dt"])
    res = []
    for s in splits:
        rdm = extract_rdm(filename, ix=s, est_type=est_type, rdm_type=rdm_type)
        res.append(analyse_split(one_body, rdm))

    es, errs = zip(*res)
    data = pd.DataFrame(
        {"tau": numpy.array(splits) * dt, "OneBody": es, "OneBody_error": errs}
    )
    return data
