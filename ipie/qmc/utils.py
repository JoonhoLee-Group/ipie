import numpy


def set_rng_seed(seed, comm, gpu=False):
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if comm.rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype="i4")
            # Can't directly json serialise numpy arrays
        else:
            seed = numpy.empty(1, dtype="i4")
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + comm.rank
    if gpu:
        import cupy

        cupy.random.seed(seed)
    else:
        numpy.random.seed(seed)
    if comm.rank == 0:
        print("# random seed is {}".format(seed))
    return seed


def gpu_synchronize(gpu):
    if gpu:
        import cupy
        cupy.cuda.stream.get_current_stream().synchronize()

    return
