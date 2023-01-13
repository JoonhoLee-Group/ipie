
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

#!/usr/bin/env python
"""Run a reblocking analysis on ipie QMC output files."""

import glob
import json
import warnings

import h5py
import numpy
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pyblock

import scipy.stats

from ipie.analysis.autocorr import reblock_by_autocorr
from ipie.analysis.extraction import (extract_data, extract_mixed_estimates,
                                      extract_rdm, get_metadata, set_info,
                                      extract_observable,
                                      extract_data_from_textfile)
from ipie.utils.linalg import get_ortho_ao_mod
from ipie.utils.misc import get_from_dict


def average_single(frame, delete=True, multi_sym=False):
    if multi_sym:
        short = frame.groupby("Iteration")
    else:
        short = frame
    means = short.mean()
    err = short.aggregate(lambda x: scipy.stats.sem(x, ddof=1))
    averaged = means.merge(
        err, left_index=True, right_index=True, suffixes=("", "_error")
    )
    columns = [c for c in averaged.columns.values if "_error" not in c]
    columns = [[c, c + "_error"] for c in columns]
    columns = [item for sublist in columns for item in sublist]
    averaged.reset_index(inplace=True)
    delcol = ["Weight", "Weight_error"]
    for d in delcol:
        if delete:
            columns.remove(d)
    return averaged[columns]


def average_ratio(numerator, denominator):
    re_num = numerator.real
    re_den = denominator.real
    im_num = numerator.imag
    im_den = denominator.imag
    # When doing FP we need to compute E = \bar{ENumer} / \bar{EDenom}
    # Only compute real part of the energy
    num_av = re_num.mean() * re_den.mean() + im_num.mean() * im_den.mean()
    den_av = re_den.mean() ** 2 + im_den.mean() ** 2
    mean = num_av / den_av
    # Doing error analysis properly is complicated. This is not correct.
    re_nume = scipy.stats.sem(re_num)
    re_dene = scipy.stats.sem(re_den)
    # Ignoring the fact that the mean includes complex components.
    cov = numpy.cov(re_num, re_den)[0, 1]
    nsmpl = len(re_num)
    error = (
        abs(mean)
        * (
            (re_nume / re_num.mean()) ** 2
            + (re_dene / re_den.mean()) ** 2
            - 2 * cov / (nsmpl * re_num.mean() * re_den.mean())
        )
        ** 0.5
    )

    return (mean, error)


def average_fp(frame):
    iteration = numpy.real(frame["Iteration"].values)
    frame = frame.drop("Iteration", axis=1)
    real_df = frame.apply(lambda x: x.real)
    imag_df = frame.apply(lambda x: x.imag)
    real_df["Iteration"] = iteration
    imag_df["Iteration"] = iteration
    real = average_single(real_df, multi_sym=True)
    imag = average_single(imag_df, multi_sym=True)
    results = pd.DataFrame()
    re_num = real.ENumer.values
    re_den = real.EDenom.values
    im_num = imag.ENumer.values
    im_den = imag.EDenom.values
    results["Iteration"] = sorted(real_df.groupby("Iteration").groups.keys())
    # When doing FP we need to compute E = \bar{ENumer} / \bar{EDenom}
    # Only compute real part of the energy
    results["E"] = (re_num * re_den + im_num * im_den) / (re_den**2 + im_den**2)
    # Doing error analysis properly is complicated. This is not correct.
    re_nume = real.ENumer_error.values
    re_dene = real.EDenom_error.values
    # Ignoring the fact that the mean includes complex components.
    cov_nd = (
        real_df.groupby("Iteration")
        .apply(lambda x: x["ENumer"].cov(x["EDenom"]))
        .values
    )
    nsamp = len(re_nume)
    results["E_error"] = (
        numpy.abs(results.E)
        * (
            (re_nume / re_num) ** 2
            + (re_dene / re_den) ** 2
            - 2 * cov_nd / (nsamp * re_num * re_den)
        )
        ** 0.5
    )
    return results


def reblock_mixed(groupby, columns, verbose=False):
    analysed = []
    for group, frame in groupby:
        drop = [
            "index",
            "Time",
            "EDenom",
            "ENumer",
            "Weight",
            "Overlap",
            "WeightFactor",
            "EHybrid",
        ]
        if not verbose:
            drop += ["E1Body", "E2Body"]
        short = frame.reset_index()
        try:
            short = short.drop(columns + drop, axis=1)
        except KeyError:
            short = short.drop(columns + ["index"], axis=1)

        (data_len, blocked_data, covariance) = pyblock.pd_utils.reblock(short)
        reblocked = pd.DataFrame({"ETotal": [0.0]})
        for c in short.columns:
            try:
                rb = pyblock.pd_utils.reblock_summary(blocked_data.loc[:, c])
                reblocked[c] = rb["mean"].values[0]
                reblocked[c + "_error"] = rb["standard error"].values
                reblocked[c + "_error_error"] = rb["standard error error"].values
                ix = list(blocked_data[c]["optimal block"]).index("<---    ")
                reblocked[c + "_nsamp"] = data_len.values[ix]
            except KeyError:
                if verbose:
                    print(
                        "Reblocking of {:4} failed. Insufficient "
                        "statistics.".format(c)
                    )
        for i, v in enumerate(group):
            reblocked[columns[i]] = v
        analysed.append(reblocked)

    final = pd.concat(analysed, sort=True)

    y = short["ETotal"].values
    reblocked_ac = reblock_by_autocorr(y, verbose=verbose)
    for c in reblocked_ac.columns:
        final[c] = reblocked_ac[c].values

    return final


def reblock_free_projection(frame):
    short = frame.drop(["Time", "Weight", "ETotal"], axis=1)
    analysed = []
    (data_len, blocked_data, covariance) = pyblock.pd_utils.reblock(short)
    reblocked = pd.DataFrame()
    denom = blocked_data.loc[:, "EDenom"]
    for c in short.columns:
        if c != "EDenom":
            nume = blocked_data.loc[:, c]
            cov = covariance.xs("EDenom", level=1)[c]
            ratio = pyblock.error.ratio(nume, denom, cov, data_len)
            rb = pyblock.pd_utils.reblock_summary(ratio)
            try:
                if c == "ENumer":
                    c = "ETotal"
                reblocked[c] = rb["mean"].values
                reblocked[c + "_error"] = rb["standard error"].values
            except KeyError:
                print(
                    "Reblocking of {:4} failed. Insufficient " "statistics.".format(c)
                )
    analysed.append(reblocked)

    if len(analysed) == 0:
        return None
    else:
        return pd.concat(analysed)


def reblock_local_energy(filename, skip=0):
    data = extract_mixed_estimates(filename)
    results = reblock_mixed(data.apply(numpy.real)[skip:])
    if results is None:
        return None
    else:
        try:
            energy = results["ETotal"].values[0]
            error = results["ETotal_error"].values[0]
            return (energy, error)
        except KeyError:
            return None


def average_rdm(files, skip=1, est_type="back_propagated", rdm_type="one_rdm", ix=None):

    rdm_series = extract_rdm(files, est_type=est_type, rdm_type=rdm_type, ix=ix)
    rdm_av = rdm_series[skip:].mean(axis=0)
    rdm_err = rdm_series[skip:].std(axis=0, ddof=1) / len(rdm_series) ** 0.5
    return rdm_av, rdm_err


def average_correlation(gf):
    ni = numpy.diagonal(gf, axis1=2, axis2=3)
    mg = gf.mean(axis=0)
    hole = 1.0 - numpy.sum(ni, axis=1)
    hole_err = hole.std(axis=0, ddof=1) / len(hole) ** 0.5
    spin = 0.5 * (ni[:, 0, :] - ni[:, 1, :])
    spin_err = spin.std(axis=0, ddof=1) / len(hole) ** 0.5
    return (hole.mean(axis=0), hole_err, spin.mean(axis=0), spin_err, gf)


def average_tau(frames):

    data_len = frames.size()
    means = frames.mean()
    err = numpy.sqrt(frames.var())
    covs = frames.cov().loc[:, "ENumer"].loc[:, "EDenom"]
    energy = means["ENumer"] / means["EDenom"]
    sqrtn = numpy.sqrt(data_len)
    energy_err = (
        (err["ENumer"] / means["ENumer"]) ** 2.0
        + (err["EDenom"] / means["EDenom"]) ** 2.0
        - 2 * covs / (means["ENumer"] * means["EDenom"])
    ) ** 0.5

    energy_err = abs(energy / sqrtn) * energy_err
    eproj = means["ETotal"]
    eproj_err = err["ETotal"] / numpy.sqrt(data_len)
    weight = means["Weight"]
    weight_error = err["Weight"]
    numerator = means["ENumer"]
    numerator_error = err["ENumer"]
    results = pd.DataFrame(
        {
            "ETotal": energy,
            "ETotal_error": energy_err,
            "Eproj": eproj,
            "Eproj_error": eproj_err,
            "weight": weight,
            "weight_error": weight_error,
            "numerator": numerator,
            "numerator_error": numerator_error,
        }
    ).reset_index()

    return results


def analyse_back_propagation(frames):
    frames[["E", "E1b", "E2b"]] = frames[["E", "E1b", "E2b"]]
    frames = frames.apply(numpy.real)
    frames = frames.groupby(["nbp", "dt"])
    data_len = frames.size()
    means = frames.mean().reset_index()
    # calculate standard error of the mean for grouped objects. ddof does
    # default to 1 for scipy but it's different elsewhere, so let's be careful.
    errs = frames.aggregate(lambda x: scipy.stats.sem(x, ddof=1)).reset_index()
    full = pd.merge(means, errs, on=["nbp", "dt"], suffixes=("", "_error"))
    columns = full.columns.values[2:]
    columns = numpy.insert(columns, 0, "nbp")
    columns = numpy.insert(columns, 1, "dt")
    return full[columns]


def analyse_itcf(itcf):
    means = itcf.mean(axis=(0, 1), dtype=numpy.float64)
    n = itcf.shape[0] * itcf.shape[1]
    errs = itcf.std(axis=(0, 1), ddof=1, dtype=numpy.float64) / numpy.sqrt(n)
    return (means, errs)


def analyse_simple(files, start_time):
    data = ipie.analysis.extraction.extract_hdf5_data_sets(files)
    norm_data = []
    for (g, f) in zip(data, files):
        (m, norm, bp, itcf, itcfk, mixed_rdm, bp_rdm) = g
        dt = m.get("qmc").get("dt")
        free_projection = m.get("propagators").get("free_projection")
        step = m.get("qmc").get("nmeasure")
        read_rs = m.get("psi").get("read_file") is not None
        nzero = numpy.nonzero(norm["Weight"].values)[0][-1]
        start = int(start_time / (step * dt)) + 1
        if read_rs:
            start = 0
        if free_projection:
            reblocked = average_fp(norm[start:nzero])
        else:
            reblocked = reblock_mixed(norm[start:nzero].apply(numpy.real))
            columns = ipie.analysis.extraction.set_info(reblocked, m)
        norm_data.append(reblocked)
    return pd.concat(norm_data)


def analyse_back_prop(files, start_time):
    full = []
    for f in files:
        md = get_metadata(f)
        step = get_from_dict(md, ["qmc", "nmeasure"])
        dt = get_from_dict(md, ["qmc", "dt"])
        tbp = get_from_dict(md, ["estimators", "estimators", "back_prop", "tau_bp"])
        start = min(1, int(start_time / tbp) + 1)
        data = extract_data(f, "back_propagated", "energies")[start:]
        av = data.mean().to_frame().T
        err = (data.std() / len(data) ** 0.5).to_frame().T
        res = pd.merge(
            av, err, left_index=True, right_index=True, suffixes=("", "_error")
        )
        columns = set_info(res, md)
        full.append(res)
    return pd.concat(full).sort_values("tau_bp")

def reblock_minimal(files, start_block=0, verbose=False):
    """Minimal blocking analysis using approximate autocorrelation time.

    Parses from textfile.
    """
    reblocked = []
    if isinstance(files, str):
        _files = [files]
    else:
        _files = files
    for f in _files:
        if '.h5' in f:
            data = extract_observable(f)[start_block:]
        else:
            data = extract_data_from_textfile(f)[start_block:]
        y = data["ETotal"].values
        rb = reblock_by_autocorr(y, verbose=verbose)
        rb['filename'] = f
        reblocked.append(rb)
    df = pd.concat(reblocked)
    return df


def analyse_estimates(files, start_time, multi_sim=False, av_tau=False, verbose=False):
    mds = []
    basic = []
    if av_tau:
        data = []
        for f in files:
            data.append(extract_mixed_estimates(f))
        full = pd.concat(data).groupby("Iteration")
        av = average_tau(full)
        print(av.apply(numpy.real).to_string())
    else:
        for f in files:
            md = get_metadata(f)
            read_rs = get_from_dict(md, ["psi", "read_rs"])
            step = get_from_dict(md, ["qmc", "nsteps"])
            dt = get_from_dict(md, ["qmc", "dt"])
            fp = get_from_dict(md, ["propagators", "free_projection"])
            start = int(start_time / (step * dt)) + 1
            if read_rs:
                start = 0
            data = extract_mixed_estimates(f, start)
            columns = set_info(data, md)
            basic.append(data.drop("Iteration", axis=1))
            mds.append(md)

        new_columns = []
        for c in columns:
            if c != "E_T":
                new_columns += [c]
        columns = new_columns

        basic = pd.concat(basic).groupby(columns)
        if fp:
            basic_av = reblock_free_projection(basic, columns)
        else:
            basic_av = reblock_mixed(basic, columns, verbose=verbose)

        base = files[0].split("/")[-1]
        outfile = "analysed_" + base
        with h5py.File(outfile, "w") as fh5:
            fh5["metadata"] = numpy.array(mds).astype("S")
            try:
                fh5["basic/estimates"] = basic_av.drop(
                    "integrals", axis=1
                ).values.astype(float)
            except KeyError:
                pass
            fh5["basic/headers"] = numpy.array(basic_av.columns.values).astype("S")

    return basic_av


def analyse_ekt_ipea(filename, ix=None, cutoff=1e-14, screen_factor=1):
    rdm, rdm_err = average_rdm(filename, rdm_type="one_rdm", ix=ix)
    fock_1h_av, fock_1h_err = average_rdm(filename, rdm_type="fock_1h", ix=ix)
    fock_1p_av, fock_1p_err = average_rdm(filename, rdm_type="fock_1p", ix=ix)
    rdm[numpy.abs(rdm) < screen_factor * rdm_err] = 0.0
    fock_1h_av[numpy.abs(fock_1h_av) < screen_factor * fock_1h_err] = 0.0
    fock_1p_av[numpy.abs(fock_1p_av) < screen_factor * fock_1p_err] = 0.0
    # Spin average
    rdm = rdm[0] + rdm[1]
    rdm = 0.5 * numpy.real(rdm + rdm.conj().T)
    rdm1_reg, X = get_ortho_ao_mod(rdm, LINDEP_CUTOFF=cutoff)
    # 1-hole / IP
    fockT = numpy.dot(X.conj().T, numpy.dot(fock_1h_av, X))
    eip, eip_vec = numpy.linalg.eigh(fockT)
    norb = rdm.shape[-1]
    I = numpy.eye(norb)
    gamma = 2.0 * I - rdm.T
    gamma_reg, X = get_ortho_ao_mod(gamma, LINDEP_CUTOFF=cutoff)
    fockT = numpy.dot(X.conj().T, numpy.dot(fock_1p_av, X))
    eea, eea_vec = numpy.linalg.eigh(fockT)
    return (eip, eip_vec), (eea, eea_vec)
