
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
        import cupy
        return cupy.asnumpy(obj)
    else:
        return obj


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
        sha1 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src,
                                       stderr=subprocess.DEVNULL).strip()
        suffix = subprocess.check_output(
            ["git", "status", "-uno", "--porcelain", "./ipie"], cwd=src
        ).strip()
        local_mods = subprocess.check_output(
            ["git", "status", "--porcelain", "./ipie"], cwd=src
        ).strip().decode('utf-8').split()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src
        ).strip()
    except subprocess.CalledProcessError as e:
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
        ("__dict__") in dir(obj)
        and not isinstance(obj, types.FunctionType)
        and not is_h5file(obj)
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
        print("# Initial {} upper_bound is {}".format(name, upper_bound))
        print("# Initial {} lower_bound is {}".format(name, lower_bound))
        print("# Adjusted {} size is {}".format(name, stack_size))
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
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
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
    print("# ipie version: {:s}".format(ipie.__version__))
    if sha1 is not None:
        print("# Git hash: {:s}.".format(sha1))
        print("# Git branch: {:s}.".format(branch))
    if len(local_mods)  > 0:
        print("# Found uncommitted changes and/or untracked files.")
        for prefix, file in zip(local_mods[::2], local_mods[1::2]):
            if prefix == 'M':
                print("# Modified : {:s}".format(file))
            elif prefix == '??':
                print("# Untracked : {:s}".format(file))
    print("# Calculation uuid: {:s}.".format(uuid))
    mem = get_node_mem()
    print("# Approximate memory available per node: {:.4f} GB.".format(mem))
    print("# Running on {:d} MPI rank{:s}.".format(nranks, "s" if nranks > 1 else ""))
    hostname = socket.gethostname()
    print("# Root processor name: {}".format(hostname))
    py_ver = sys.version.splitlines()
    print("# Python interpreter: {:s}".format(" ".join(py_ver)))
    info = {"nranks": nranks, "python": py_ver, "branch": branch, "sha1": sha1}
    from importlib import import_module

    for lib in ["numpy", "scipy", "h5py", "mpi4py", "cupy"]:
        try:
            l = import_module(lib)
            # Strip __init__.py
            path = l.__file__[:-12]
            vers = l.__version__
            print("# Using {:s} v{:s} from: {:s}.".format(lib, vers, path))
            info["{:s}".format(lib)] = {"version": vers, "path": path}
            if lib == "numpy":
                try:
                    np_lib = l.__config__.blas_opt_info["libraries"]
                except AttributeError:
                    np_lib = l.__config__.blas_ilp64_opt_info["libraries"]
                print("# - BLAS lib: {:s}".format(" ".join(np_lib)))
                try:
                    lib_dir = l.__config__.blas_opt_info["library_dirs"]
                except AttributeError:
                    lib_dir = l.__config__.blas_ilp64_opt_info["library_dirs"]
                print("# - BLAS dir: {:s}".format(" ".join(lib_dir)))
                info["{:s}".format(lib)]["BLAS"] = {
                    "lib": " ".join(np_lib),
                    "path": " ".join(lib_dir),
                }
            elif lib == "mpi4py":
                mpicc = l.get_config().get("mpicc", "none")
                print("# - mpicc: {:s}".format(mpicc))
                info["{:s}".format(lib)]["mpicc"] = mpicc
            elif lib == "cupy":
                try:
                    cu_info = l.cuda.runtime.getDeviceProperties(0)
                    cuda_compute = l.cuda.Device().compute_capability
                    cuda_version = str(l.cuda.runtime.runtimeGetVersion())
                    cuda_compute = cuda_compute[0] + "." + cuda_compute[1]
                    # info['{:s}'.format(lib)]['cuda'] = {'info': ' '.join(np_lib),
                    #                                    'path': ' '.join(lib_dir)}
                    version_string = (
                        cuda_version[:2]
                        + "."
                        + cuda_version[2:4]
                        + "."
                        + cuda_version[4]
                    )
                    print("# - CUDA compute capability: {:s}".format(cuda_compute))
                    print("# - CUDA version: {}".format(version_string))
                    print("# - GPU Type: {:s}".format(str(cu_info["name"])[1:]))
                    print(
                        "# - GPU Mem: {:.3f} GB".format(
                            cu_info["totalGlobalMem"] / (1024**3.0)
                        )
                    )
                    print(
                        "# - Number of GPUs: {:d}".format(
                            l.cuda.runtime.getDeviceCount()
                        )
                    )
                except:
                    print("# cupy import error")
        except (ModuleNotFoundError, ImportError):
            print("# Package {:s} not found.".format(lib))
    return info


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(" # Time : {} ".format(end - start))
        return res

    return wrapper
