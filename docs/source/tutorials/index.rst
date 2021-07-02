Tutorials
=========

Here we'll outline how use pauxy to calculate the imaginary time correlation function
(ITCF) as well as various ground state estimates of the 3x3 Hubbard model with periodic
boundary conditions.

The input file is given as follows:

.. literalinclude:: calcs/hubbard/input.json


Note that we added a small twist to lift the degeneracy in the free electron ground state,
we're using a free-electron trial wavefunction and that to calculate the ITCF we also need
to set the back propagation time to be sufficiently large so that the left handed
wavefunction is close to the ground state.

First we'll run pauxy as follows:

.. code-block:: bash

    $ python3 -u ~/path/to/pauxy/bin/pauxy input.json > 3x3.out

By default pauxy will print some basic information to stdout, including periodic estimates
for mixed estimates of the energy. All estimates will also be saved (by default) to the
estimates.0.h5 file.

Once the calculation has finished we can inspect the output. It's helpful to first plot
the energy data from the 3x3.out to inspect when the simulation has equilibrated and to
get a rough idea of the quality of the result.

.. plot::

    import pandas as pd
    import matplotlib.pyplot as pl

    data = pd.read_csv('calcs/hubbard/3x3.out', sep=r'\s+', comment='#')
    dt = 0.05
    pl.plot(data.iteration*dt, data.E)
    pl.xlabel(r'$\tau t')
    pl.xlabel(r'Energy (t)')

To be safe we'll discard data before :math:`\tau t=10` and analyse the data as follows:

.. code-block:: bash

    $ ~/path/to/pauxy/tools/reblock.py -s 10 -f estimates.0.h5

which will analyse all the estimates in estimates.0.h5 and dump the results to a second
file called analysed_estimates.h5. We can extract then extract the estimates by doing

.. code-block:: bash

    $ ~/path/to/pauxy/tools/extract_observable.py -o 'energy' > basic.out


.. literalinclude:: calcs/hubbard/basic.out

which will extract the mixed estimates. For the back propagated estimates we can do

.. code-block:: bash

    $ ~/path/to/pauxy/tools/extract_observable.py -o 'back_propagated' > back_propagated.out

to find

.. literalinclude:: calcs/hubbard/back_propagated.out

and finally the greater one-particle imaginary time green's function as

.. code-block:: bash

    $ ~/path/to/pauxy/tools/extract_observable.py -o 'itcf' -s up -t greater -e 0,0 -f analysed_observables.h5 > itcf.out

which we can then plot

.. plot::

    import pandas as pd
    import matplotlib.pyplot as pl

    data = pd.read_csv('calcs/hubbard/itcf.out', sep=r'\s+', comment='#')
    pl.errorbar(data.tau, data.G_greater_spin_up_00,
                yerr=data.G_greater_spin_up_00_err, fmt='o')
    pl.xlabel(r'$\tau t$')
    pl.ylabel(r'$G_{00}^{\uparrow\uparrow}(\tau)$')

As you can see, the error bars are still quite large so you should run for longer.

.. toctree::
    :maxdepth: 2
    :glob:
