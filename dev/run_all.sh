#!/bin/bash
# pylint -s no --disable=R,C ipie/
# black --check ipie
# pytest
# mpiexec -n 2 python -m pytest ipie/legacy/walkers/tests/test_handler.py
# mpiexec -n 6 python -m pytest ipie/estimators/tests/test_generic_chunked.py
# mpiexec -n 6 python -m pytest ipie/propagation/tests/test_generic_chunked.py
#mpiexec -np 4 python -u -m pytest -sv --with-mpi ipie/qmc/tests/test_mpi_integration.py
./tools/run_examples.sh
