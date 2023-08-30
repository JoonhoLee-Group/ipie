import sys


from ipie.thermal.system.ueg import UEG
from ipie.systems.generic import Generic


def get_system(sys_opts=None, verbose=0, comm=None):
    """Wrapper to select system class

    Parameters
    ----------
    sys_opts : dict
        System input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    system : object
        System class.
    """
    if not ("name" in sys_opts):
        sys_opts["name"] = "Generic"

    if sys_opts["name"] == "UEG":
        system = UEG(sys_opts, verbose)
    elif (
        sys_opts["name"] == "Hubbard"
        or sys_opts["name"] == "HubbardHolstein"
        or sys_opts["name"] == "Generic"
    ):
        nup, ndown = sys_opts.get("nup"), sys_opts.get("ndown")
        if nup is None or ndown is None:
            if comm.rank == 0:
                print("# Error: Number of electrons not specified.")
                sys.exit()
        nelec = (nup, ndown)
        system = Generic(nelec, verbose)
    else:
        if comm.rank == 0:
            print(f"# Error: unrecognized system name {sys_opts['name']}.")
            sys.exit()

    return system
