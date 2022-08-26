Developer Guidelines
====================

Code Format
-----------

* Use black.

GPU
---

* Due to current global config setup for device selection gpu unit tests need to be
  run separately from the bulk of the unit tests.
* This is achieved by marking the unit test like so:

.. codeblock:: python

    @pytest.mark.gpu
    def test_my_special_test():

* Currently it is advised to place gpu specific unit tests in filenames with _gpu.py in
  the name.
* To run the tests use:

.. codeblock:: bash

     export IPIE_USE_GPU=1; mpirun -np 1 pytest -m gpu -sv

* Note if running CPU test afterwards it may be necessary to clear the environment
  variable!
