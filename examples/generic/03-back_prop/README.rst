Back Propagation
================

In this example we will outline how to compute the back propagated 1RDM.

Generate the integrals as before but use the `-b` option to generate a skeleton back
propagated estimator block.

.. code-block:: json

    {
        "system": {
            "name": "Generic",
            "nup": 5,
            "ndown": 5,
            "integrals": "afqmc.h5"
        },
        "qmc": {
            "dt": 0.005,
            "nsteps": 5000,
            "nmeasure": 10,
            "nwalkers": 30,
            "pop_control": 1
        },
        "trial": {
            "filename": "afqmc.h5"
        },
        "estimators": {
            "back_propagated": {
                "tau_bp": 2.0,
                "nsplit": 4
            }
        }
    }

In the above example the back propagated one-rdm will be computed using a propagation time
of 2 au. The path length is split into four (nsplit) lengths (0.5, 1.0, 1.5, 2.0) in order
to check the convergence wrt back propagation time.

The rdm can be extracted from the estimates files using

.. code-block:: python

    from pauxy.analysis.extraction import extract_rdm

    # By default the longest path length is selected
    rdm = extract_rdm('estimates.0.h5')

    # can provide an index to get an rdm at a specific split
    rdm_0 = extract_rdm('estimates.0.h5', ix=100) # tau = 0.5
    rdm_1 = extract_rdm('estimates.0.h5', ix=200) # tau = 1.0
    rdm_1 = extract_rdm('estimates.0.h5', ix=300) # tau = 1.5

    # To get the averaged rdm use
    from pauxy.analysis.blocking import average_rdm
    rdm_av, rdm_err = average_rdm('estimates.0.h5')
