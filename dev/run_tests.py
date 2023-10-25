#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
from typing import List


def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    modes = ["pytest", "pylint", "black", "mpi", "examples", "flynt", "integration", "all"]
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in modes:
        parser.add_argument(
            f"--{opt}",
            dest=f"{opt}",
            action="store_true",
            help=f"Run {opt} tests.",
        )
    if len(args) == 0:
        parser.print_help()
        sys.exit()
    options = parser.parse_args(args)

    return options


def _run_subprocess(job_string: str, shell=False) -> int:
    try:
        if shell:
            output = subprocess.check_output(job_string, shell=shell)
        else:
            output = subprocess.check_output(job_string.split())
        print(output.decode("utf-8"))
        return 0
    except subprocess.CalledProcessError as err:
        if err.stdout is not None:
            print(err.stdout.decode("utf-8"))
        if err.stderr is not None:
            print(err.stderr.decode("utf-8"))
        return err.returncode


def run_black():
    return _run_subprocess("black --check ipie")


def run_flynt():
    return _run_subprocess("flynt --fail-on-change --dry-run ipie")


def run_pylint():
    return _run_subprocess("pylint --score n --disable=R,C ipie/")


def run_mpi():
    jobs = [
        "mpiexec -n 2 python -m pytest ipie/legacy/walkers/tests/test_handler.py",
        "mpiexec -n 6 python -m pytest ipie/estimators/tests/test_generic_chunked.py",
        "mpiexec -n 6 python -m pytest ipie/propagation/tests/test_generic_chunked.py",
    ]
    err = 0
    for job in jobs:
        print(f" - Running: {job}")
        err += _run_subprocess(job, shell=True)
    return err


def run_integration():
    return _run_subprocess(
        "mpiexec -np 4 python -u ipie/qmc/tests/test_mpi_integration.py",
        shell=True,
    )


def run_pytest():
    return _run_subprocess("pytest -n=auto -sv ipie/")


def run_examples() -> List[str]:
    examples = sorted(glob.glob("examples/*"))
    # Legacy
    legacy = {"01": "", "05": "--frozen-core 5"}
    legacy_dirs = list(legacy.keys())
    err = 0
    failed_tests = []
    for leg, arg in legacy.items():
        leg_dir = glob.glob(f"examples/{leg}-*")[0]
        print(f" - Running: {leg_dir}")
        err = _run_subprocess(f"python {leg_dir}/scf.py")
        if err > 0:
            failed_tests.append(f"{leg}-scf.py")
        err = _run_subprocess(f"python tools/pyscf/pyscf_to_ipie.py -i scf.chk {arg}")
        if err > 0:
            failed_tests.append(f"{leg}-pyscf_to_ipie.py")
        try:
            os.remove("scf.chk")
        except FileNotFoundError:
            ...
    for example in examples:
        if os.path.isdir(example):
            if example.split("/")[-1].split("-")[0] in legacy_dirs:
                continue
            else:
                print(f" - Running: {example}")
                err = _run_subprocess(f"python -u {example}/run_afqmc.py")
                if err > 0:
                    failed_tests.append(f"{example}-run_afqmc.py")

    return failed_tests


run_test = {
    "black": run_black,
    "pylint": run_pylint,
    "flynt": run_flynt,
    "mpi": run_mpi,
    "integration": run_integration,
    "pytest": run_pytest,
    "examples": run_examples,
}


def main(args):
    options = parse_args(args)
    err = 0
    run_all = options.all
    for opt, val in vars(options).items():
        if opt == "all":
            continue
        if val or run_all:
            print(f"Running: {opt}")
            val = run_test[opt]()
            if isinstance(val, list):
                for x in val:
                    print(x)
                val = len(val)
            err += val
            if val:
                print(f"Failed: {opt}.")
            else:
                print(f"Passed: {opt}")
    if err == 0:
        print("All passed! phew")
    return err


if __name__ == "__main__":
    err = main(sys.argv[1:])
    sys.exit(err)
