import numpy

from ipie.utils.misc import is_cupy, to_numpy


def construct_force_bias_batch(hamiltonian, walker_batch, trial, mpi_handler=None):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if walker_batch.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater":
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walker_batch, trial, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(
                hamiltonian, walker_batch, trial
            )
    elif (
        walker_batch.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater"
    ):
        return construct_force_bias_batch_multi_det_trial(
            hamiltonian, walker_batch, trial
        )


def construct_force_bias_batch_multi_det_trial(hamiltonian, walker_batch, trial):
    Ga = walker_batch.Ga.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    Gb = walker_batch.Gb.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol_vecs):
        vbias_batch = numpy.empty(
            (hamiltonian.nchol, walker_batch.nwalkers), dtype=numpy.complex128
        )
        vbias_batch.real = hamiltonian.chol_vecs.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol_vecs.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol_vecs.T.dot(Ga.T + Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
        return vbias_batch_tmp


def construct_force_bias_batch_single_det(hamiltonian, walker_batch, trial):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if is_cupy(
        trial.psi
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        assert cupy.is_available()
        isrealobj = cupy.isrealobj
        empty = cupy.empty
    else:
        isrealobj = numpy.isrealobj
        empty = numpy.empty

    if walker_batch.rhf:
        Ghalfa = walker_batch.Ghalfa.reshape(
            walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
        )
        if isrealobj(trial._rchola) and isrealobj(trial._rcholb):
            vbias_batch_real = 2.0 * trial._rchola.dot(Ghalfa.T.real)
            vbias_batch_imag = 2.0 * trial._rchola.dot(Ghalfa.T.imag)
            vbias_batch = empty(
                (walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype
            )
            vbias_batch.real = vbias_batch_real.T.copy()
            vbias_batch.imag = vbias_batch_imag.T.copy()
            if is_cupy(trial.psi):
                import cupy

                cupy.cuda.stream.get_current_stream().synchronize()
            return vbias_batch
        else:
            vbias_batch_tmp = 2.0 * trial._rchola.dot(Ghalfa.T)
            if is_cupy(trial.psi):
                import cupy

                cupy.cuda.stream.get_current_stream().synchronize()

            return vbias_batch_tmp.T

    else:
        if trial.mixed_precision:
            dGhalfa = (
                walker_batch.Ghalfa.reshape(
                    walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
                )
                - trial.psia.T.ravel()
            )
            dGhalfb = (
                walker_batch.Ghalfb.reshape(
                    walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
                )
                - trial.psib.T.ravel()
            )
            dGhalfa = dGhalfa.astype(numpy.complex64)
            dGhalfb = dGhalfb.astype(numpy.complex64)
            if isrealobj(trial._rchola) and isrealobj(trial._rcholb):
                # single precision
                vbias_batch_real = trial._rchola.dot(
                    dGhalfa.T.real
                ) + trial._rcholb.dot(dGhalfb.T.real)
                vbias_batch_imag = trial._rchola.dot(
                    dGhalfa.T.imag
                ) + trial._rcholb.dot(dGhalfb.T.imag)
                # double precision
                vbias_batch = empty(
                    (walker_batch.nwalkers, hamiltonian.nchol),
                    dtype=walker_batch.Ghalfa.dtype,
                )
                vbias_batch.real = vbias_batch_real.T.copy() + trial._vbias0.real
                vbias_batch.imag = vbias_batch_imag.T.copy() + trial._vbias0.imag
                if is_cupy(trial.psi):
                    import cupy

                    cupy.cuda.stream.get_current_stream().synchronize()

                return vbias_batch
            else:
                vbias_batch_tmp = trial._rchola.dot(dGhalfa.T) + trial._rcholb.dot(
                    dGhalfb.T
                )
                vbias_batch_tmp += trial._vbias0
                if is_cupy(trial.psi):
                    import cupy

                    cupy.cuda.stream.get_current_stream().synchronize()

                return vbias_batch_tmp.T
        else:
            Ghalfa = walker_batch.Ghalfa.reshape(
                walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
            )
            Ghalfb = walker_batch.Ghalfb.reshape(
                walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
            )
            if isrealobj(trial._rchola) and isrealobj(trial._rcholb):
                vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(
                    Ghalfb.T.real
                )
                vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(
                    Ghalfb.T.imag
                )
                vbias_batch = empty(
                    (walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype
                )
                vbias_batch.real = vbias_batch_real.T.copy()
                vbias_batch.imag = vbias_batch_imag.T.copy()
                if is_cupy(trial.psi):
                    import cupy

                    cupy.cuda.stream.get_current_stream().synchronize()

                return vbias_batch
            else:
                vbias_batch_tmp = trial._rchola.dot(Ghalfa.T) + trial._rcholb.dot(
                    Ghalfb.T
                )
                if is_cupy(trial.psi):
                    import cupy

                    cupy.cuda.stream.get_current_stream().synchronize()

                return vbias_batch_tmp.T


def construct_force_bias_batch_single_det_chunked(
    hamiltonian, walker_batch, trial, handler
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    do_gpu = is_cupy(trial.psi) # if even one array is a cupy array we should assume the rest is done with cupy

    if do_gpu:
        import cupy
        assert cupy.is_available()
        isrealobj = cupy.isrealobj
        empty = cupy.empty
        zeros_like = cupy.zeros_like
        zeros = cupy.zeros
    else:
        isrealobj = numpy.isrealobj
        empty = numpy.empty
        zeros_like = numpy.zeros_like
        zeros = numpy.zeros
    where = numpy.where

    assert hamiltonian.chunked
    assert isrealobj(trial._rchola)

    Ghalfa = walker_batch.Ghalfa.reshape(
        walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
    )
    Ghalfb = walker_batch.Ghalfb.reshape(
        walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
    )

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = zeros_like(Ghalfa)
    Ghalfb_recv = zeros_like(Ghalfb)

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    ssize = handler.scomm.size
    srank = handler.scomm.rank

    vbias_batch_real_recv = zeros((hamiltonian.nchol, walker_batch.nwalkers))
    vbias_batch_imag_recv = zeros((hamiltonian.nchol, walker_batch.nwalkers))

    vbias_batch_real_send = zeros((hamiltonian.nchol, walker_batch.nwalkers))
    vbias_batch_imag_send = zeros((hamiltonian.nchol, walker_batch.nwalkers))

    vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.real
    ) + trial._rcholb_chunk.dot(Ghalfb.T.real)
    vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.imag
    ) + trial._rcholb_chunk.dot(Ghalfb.T.imag)

    senders = handler.senders
    receivers = handler.receivers
    for icycle in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if srank == isend:
                if do_gpu:
                    Ghalfa_send = cupy.asnumpy(Ghalfa_send)
                    Ghalfb_send = cupy.asnumpy(Ghalfb_send)
                    vbias_batch_real_send = cupy.asnumpy(vbias_batch_real_send)
                    vbias_batch_imag_send = cupy.asnumpy(vbias_batch_imag_send)
                handler.scomm.Send(Ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(Ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(vbias_batch_real_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(vbias_batch_imag_send, dest=receivers[isend], tag=4)
                if do_gpu:
                    Ghalfa_send = cupy.asarray(Ghalfa_send)
                    Ghalfb_send = cupy.asarray(Ghalfb_send)
                    vbias_batch_real_send = cupy.asarray(vbias_batch_real_send)
                    vbias_batch_imag_send = cupy.asarray(vbias_batch_imag_send)
            elif srank == receivers[isend]:
                sender = where(receivers == srank)[0]
                if do_gpu:
                    Ghalfa_recv = cupy.asnumpy(Ghalfa_recv)
                    Ghalfb_recv = cupy.asnumpy(Ghalfb_recv)
                    vbias_batch_real_recv = cupy.asnumpy(vbias_batch_real_recv)
                    vbias_batch_imag_recv = cupy.asnumpy(vbias_batch_imag_recv)
                handler.scomm.Recv(Ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(Ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(vbias_batch_real_recv, source=sender, tag=3)
                handler.scomm.Recv(vbias_batch_imag_recv, source=sender, tag=4)
                if do_gpu:
                    Ghalfa_recv = cupy.asarray(Ghalfa_recv)
                    Ghalfb_recv = cupy.asarray(Ghalfb_recv)
                    vbias_batch_real_recv = cupy.asarray(vbias_batch_real_recv)
                    vbias_batch_imag_recv = cupy.asarray(vbias_batch_imag_recv)
        handler.scomm.barrier()

        # prepare sending
        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.real
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.real)
        vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.imag
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.imag)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    for isend, sender in enumerate(senders):
        if handler.scomm.rank == sender:  # sending 1 xshifted to 0 xshifted_buf
            if do_gpu:
                vbias_batch_real_send = cupy.asnumpy(vbias_batch_real_send)
                vbias_batch_imag_send = cupy.asnumpy(vbias_batch_imag_send)
            handler.scomm.Send(vbias_batch_real_send, dest=receivers[isend], tag=1)
            handler.scomm.Send(vbias_batch_imag_send, dest=receivers[isend], tag=2)
            if do_gpu:
                vbias_batch_real_send = cupy.asarray(vbias_batch_real_send)
                vbias_batch_imag_send = cupy.asarray(vbias_batch_imag_send)
        elif srank == receivers[isend]:
            sender = where(receivers == srank)[0]
            if do_gpu:
                vbias_batch_real_recv = cupy.asnumpy(vbias_batch_real_recv)
                vbias_batch_imag_recv = cupy.asnumpy(vbias_batch_imag_recv)
            handler.scomm.Recv(vbias_batch_real_recv, source=sender, tag=1)
            handler.scomm.Recv(vbias_batch_imag_recv, source=sender, tag=2)
            if do_gpu:
                vbias_batch_real_recv = cupy.asarray(vbias_batch_real_recv)
                vbias_batch_imag_recv = cupy.asarray(vbias_batch_imag_recv)

    vbias_batch = empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()

    if is_cupy(trial.psi):
        import cupy

        cupy.cuda.stream.get_current_stream().synchronize()

    return vbias_batch
