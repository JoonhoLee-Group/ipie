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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

"""Various useful routines maybe not appropriate elsewhere"""

import os
import socket
import subprocess
import sys
import time
import types
from functools import reduce

import numpy
import scipy.sparse


def is_cupy(obj):
    t = str(type(obj))
    cond = "cupy" in t
    return cond


def to_numpy(obj):
    t = str(type(obj))
    cond = "cupy" in t
    if cond:
        # pylint: disable=import-error
        import cupy

        return cupy.asnumpy(obj)
    else:
        return


def get_git_info():
    """Return git info.

    Adapted from:
        http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

    Returns
    -------
    sha1 : string
        git hash with -dirty appended if uncommitted changes.
    branch : string
        Current branch
    local_mod : list of strings
        List of locally modified files tracked and untracked.
    """

    under_git = True
    try:
        src = os.path.dirname(__file__) + "/../../"
        sha1 = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=src, stderr=subprocess.DEVNULL
        ).strip()
        suffix = subprocess.check_output(
            ["git", "status", "-uno", "--porcelain", "./ipie"], cwd=src
        ).strip()
        local_mods = (
            subprocess.check_output(["git", "status", "--porcelain", "./ipie"], cwd=src)
            .strip()
            .decode("utf-8")
            .split()
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src
        ).strip()
    except subprocess.CalledProcessError:
        under_git = False
    except Exception as error:
        suffix = False
        print(f"couldn't determine git hash : {error}")
        sha1 = "none".encode()
        local_mods = []
    if under_git:
        if suffix:
            return sha1.decode("utf-8") + "-dirty", branch.decode("utf-8"), local_mods
        else:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src
            ).strip()
            return sha1.decode("utf-8"), branch.decode("utf_8"), local_mods
    else:
        return None, None, []


def is_h5file(obj):
    t = str(type(obj))
    cond = "h5py" in t
    return cond


def is_class(obj):
    cond = hasattr(obj, "__class__") and (
        ("__dict__") in dir(obj) and not isinstance(obj, types.FunctionType) and not is_h5file(obj)
    )

    return cond


def serialise(obj, verbose=0):
    obj_dict = {}
    if isinstance(obj, dict):
        items = obj.items()
    else:
        items = obj.__dict__.items()

    for k, v in items:
        if isinstance(v, scipy.sparse.csr_matrix):
            pass
        elif isinstance(v, scipy.sparse.csc_matrix):
            pass
        elif is_class(v):
            # Object
            obj_dict[k] = serialise(v, verbose)
        elif isinstance(v, dict):
            obj_dict[k] = serialise(v)
        elif isinstance(v, types.FunctionType):
            # function
            if verbose == 1:
                obj_dict[k] = str(v)
        elif hasattr(v, "__self__"):
            # unbound function
            if verbose == 1:
                obj_dict[k] = str(v)
        elif k == "estimates" or k == "global_estimates":
            pass
        elif k == "walkers":
            obj_dict[k] = [str(x) for x in v][0]
        elif isinstance(v, numpy.ndarray):
            if verbose == 3:
                if v.dtype == complex:
                    obj_dict[k] = [v.real.tolist(), v.imag.tolist()]
                else:
                    obj_dict[k] = (v.tolist(),)
            elif verbose == 2:
                if len(v.shape) == 1:
                    if v[0] is not None and v.dtype == complex:
                        obj_dict[k] = [[v.real.tolist(), v.imag.tolist()]]
                    else:
                        obj_dict[k] = (v.tolist(),)
            elif len(v.shape) == 1:
                if v[0] is not None and numpy.linalg.norm(v) > 1e-8:
                    if v.dtype == complex:
                        obj_dict[k] = [[v.real.tolist(), v.imag.tolist()]]
                    else:
                        obj_dict[k] = (v.tolist(),)
        elif k == "store":
            if verbose == 1:
                obj_dict[k] = str(v)
        elif isinstance(v, (int, float, bool, str)):
            obj_dict[k] = v
        elif isinstance(v, complex):
            obj_dict[k] = v.real
        elif v is None:
            obj_dict[k] = v
        elif is_h5file(v):
            if verbose == 1:
                obj_dict[k] = v.filename
        else:
            pass

    return obj_dict


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def update_stack(stack_size, num_slices, name="stack", verbose=False):
    lower_bound = min(stack_size, num_slices)
    upper_bound = min(stack_size, num_slices)

    while (num_slices // lower_bound) * lower_bound < num_slices:
        lower_bound -= 1
    while (num_slices // upper_bound) * upper_bound < num_slices:
        upper_bound += 1

    if (stack_size - lower_bound) <= (upper_bound - stack_size):
        stack_size = lower_bound
    else:
        stack_size = upper_bound
    if verbose:
        print(f"# Initial {name} upper_bound is {upper_bound}")
        print(f"# Initial {name} lower_bound is {lower_bound}")
        print(f"# Adjusted {name} size is {stack_size}")
    return stack_size


def print_section_header(string):
    header = """
    ################################################
    #                                              #
    #                                              #
    #                                              #
    """
    box_len = len("################################################")
    str_len = len(string)
    start = box_len // 2 - str_len // 2 - 1
    init = "#" + " " * start
    end = box_len - (box_len // 2 + str_len // 2) - 1
    fin = " " * end + "#"
    footer = """
    #                                              #
    #                                              #
    #                                              #
    ################################################
    """
    print(header + init + string + fin + footer)


def merge_dicts(a, b, path=None):
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception(f"Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = b[key]
    return a


def get_from_dict(d, k):
    """Get value from nested dictionary.

    Taken from:
        https://stackoverflow.com/questions/28225552/is-there-a-recursive-version-of-the-dict-get-built-in

    Parameters
    ----------
    d : dict
    k : list
        List specifying key to extract.

    Returns
    -------
    value : Return type or None.
    """
    try:
        return reduce(dict.get, k, d)
    except TypeError:
        # Value not found.
        return None


def get_numeric_names(d):
    names = []
    size = 0
    for k, v in d.items():
        if isinstance(v, (numpy.ndarray)):
            names.append(k)
            size += v.size
        elif isinstance(v, (int, float, complex)):
            names.append(k)
            size += 1
        elif isinstance(v, list):
            names.append(k)
            for l in v:
                if isinstance(l, (numpy.ndarray)):
                    size += l.size
                elif isinstance(l, (int, float, complex)):
                    size += 1
    return names, size


def get_node_mem():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3.0
    except:
        return 0.0


def print_env_info(sha1, branch, local_mods, uuid, nranks):
    import ipie

    version = getattr(ipie, "__version__", "Unknown")
    print(f"# ipie version: {version}")
    if sha1 is not None:
        print(f"# Git hash: {sha1:s}.")
        print(f"# Git branch: {branch:s}.")
    if len(local_mods) > 0:
        print("# Found uncommitted changes and/or untracked files.")
        for prefix, file in zip(local_mods[::2], local_mods[1::2]):
            if prefix == "M":
                print(f"# Modified : {file:s}")
            elif prefix == "??":
                print(f"# Untracked : {file:s}")
    print(f"# Calculation uuid: {uuid:s}.")
    mem = get_node_mem()
    print(f"# Approximate memory available per node: {mem:.4f} GB.")
    print(f"# Running on {nranks:d} MPI rank{'s' if nranks > 1 else '':s}.")
    hostname = socket.gethostname()
    print(f"# Root processor name: {hostname}")
    py_ver = sys.version.splitlines()
    print(f"# Python interpreter: {' '.join(py_ver):s}")
    info = {"nranks": nranks, "python": py_ver, "branch": branch, "sha1": sha1}
    from importlib import import_module

    for lib in ["numpy", "scipy", "h5py", "mpi4py", "cupy"]:
        try:
            l = import_module(lib)
            # Strip __init__.py
            path = l.__file__[:-12]
            vers = l.__version__
            print(f"# Using {lib:s} v{vers:s} from: {path:s}.")
            info[f"{lib:s}"] = {"version": vers, "path": path}
            if lib == "numpy":
                try:
                    np_lib = l.__config__.blas_opt_info["libraries"]
                except AttributeError:
                    np_lib = l.__config__.blas_ilp64_opt_info["libraries"]
                print(f"# - BLAS lib: {' '.join(np_lib):s}")
                try:
                    lib_dir = l.__config__.blas_opt_info["library_dirs"]
                except AttributeError:
                    lib_dir = l.__config__.blas_ilp64_opt_info["library_dirs"]
                print(f"# - BLAS dir: {' '.join(lib_dir):s}")
                info[f"{lib:s}"]["BLAS"] = {
                    "lib": " ".join(np_lib),
                    "path": " ".join(lib_dir),
                }
            elif lib == "mpi4py":
                mpicc = l.get_config().get("mpicc", "none")
                print(f"# - mpicc: {mpicc:s}")
                info[f"{lib:s}"]["mpicc"] = mpicc
            elif lib == "cupy":
                try:
                    cu_info = l.cuda.runtime.getDeviceProperties(0)
                    cuda_compute = l.cuda.Device().compute_capability
                    cuda_version = str(l.cuda.runtime.runtimeGetVersion())
                    cuda_compute = cuda_compute[0] + "." + cuda_compute[1]
                    # info['{:s}'.format(lib)]['cuda'] = {'info': ' '.join(np_lib),
                    #                                    'path': ' '.join(lib_dir)}
                    version_string = (
                        cuda_version[:2] + "." + cuda_version[2:4] + "." + cuda_version[4]
                    )
                    print(f"# - CUDA compute capability: {cuda_compute:s}")
                    print(f"# - CUDA version: {version_string}")
                    print(f"# - GPU Type: {str(cu_info['name'])[1:]:s}")
                    print(f"# - GPU Mem: {cu_info['totalGlobalMem'] / 1024 ** 3.0:.3f} GB")
                    print(f"# - Number of GPUs: {l.cuda.runtime.getDeviceCount():d}")
                except:
                    print("# cupy import error")
        except (ModuleNotFoundError, ImportError):
            print(f"# Package {lib:s} not found.")
    return info


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f" # Time : {end - start} ")
        return res

    return wrapper
