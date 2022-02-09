import numpy

def set_rng_seed(seed, comm):
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if comm.rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype='i4')
            # Can't directly json serialise numpy arrays
        else:
            seed = numpy.empty(1, dtype='i4')
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + comm.rank
    numpy.random.seed(seed)
    return seed
