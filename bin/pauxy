#!/usr/bin/env python3

import sys
from pauxy.qmc.calc import setup_calculation
import json


def main(input_file):
    """Simple launcher for pauxy via input file.

    Parameters
    ----------
    input_file : string
        JSON input file name.
    """
    (afqmc, comm) = setup_calculation(input_file)
    afqmc.run(comm=comm, verbose=True)
    afqmc.finalise(verbose=True)


if __name__ == '__main__':
    main(sys.argv[1])
