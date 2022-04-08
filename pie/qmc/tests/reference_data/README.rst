MPI Integration Tests Reference Data
====================================

To add a test first:

#. Create a descriptive folder name for the system / qmc algorithm in question and cd into
   it.
#. Create input.json and afqmc.h5 in wavefunction / hamiltonian needed. Your test should
   be relatively fast to run and make sure you set the `rng_seed` so the results are
   reproducible.
#. Run your example with 4 MPI tasks.
#. Run script `pie/qmc/tests/reference_data/save_benchmark.py` which should generate a file `benchmark.json`.
#. Add test directory to `pie/qmc/tests/test_pie_integration.py` list of tests.
#. Run `mpirun -np 4 python -m pytest --with-mpi test_mpi_integration.py` from the
   pie/qmc/tests directory and make sure your test passes.
#. Git add input.json, afqmc.h5 (if applicable) and benchmark.json and commit. DO NOT ADD
   estimates.*.h5.
#. Send out PR to merge new test into main.
