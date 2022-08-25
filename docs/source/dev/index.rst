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
        from ipie.config import config
        config.update_option('usef_gpu', True)

* Currently it is advised to place gpu specific unit tests in filenames with _gpu.py in
  the name.
