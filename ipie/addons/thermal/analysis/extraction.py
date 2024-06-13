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


from ipie.utils.misc import get_from_dict


def set_info(frame, md):
    ncols = len(frame.columns)
    system = md.get("system")
    hamiltonian = md.get("hamiltonian")
    trial = md.get("trial")
    qmc = md.get("params")
    fp = get_from_dict(md, ["propagators", "free_projection"])
    bp = get_from_dict(md, ["estimates", "estimates", "back_prop"])

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
