import time
import numpy
import scipy.linalg
from ipie.utils.misc import is_cupy
from ipie.utils.pack import unpack_VHS_batch

class GenericContinuous(object):
    """Propagator for generic many-electron Hamiltonian.

    Uses continuous HS transformation for exponential of two body operator.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.options.QMCOpts`
        QMC options.
    system : :class:`pie.system.System`
        System object.
    hamiltonian : :class:`pie.hamiltonian.System`
        System object.
    trial : :class:`pie.trial_wavefunctioin.Trial`
        Trial wavefunction object.
    verbose : bool
        If true print out more information during setup.
    """

    def __init__(self, system, hamiltonian, trial, qmc, options={}, verbose=False):
        
        assert(qmc.batched)

        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        start = time.time()
        if trial.ndets > 1:
            self.mf_shift = (
                    self.construct_mean_field_shift_multi_det(hamiltonian, trial)
                    )
        else:
            self.mf_shift = self.construct_mean_field_shift(hamiltonian, trial)

        if verbose:
            print("# Time to mean field shift: {} s".format(time.time()-start))
            print("# Absolute value of maximum component of mean field shift: "
                  "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift))))
        # Mean field shifted one-body propagator
        self.construct_one_body_propagator(hamiltonian, qmc.dt)
        # Constant core contribution modified by mean field shift.
        self.mf_core = hamiltonian.ecore + 0.5*numpy.dot(self.mf_shift, self.mf_shift)
        self.nstblz = qmc.nstblz

        self.nwalkers = qmc.nwalkers
        self.vbias_batch = numpy.zeros((qmc.nwalkers, hamiltonian.nfields), dtype=numpy.complex128)

        self.ebound = (2.0/self.dt)**0.5
        self.mean_local_energy = 0

        if verbose:
            print("# Finished setting up propagation.GenericContinuous.")

    def construct_mean_field_shift(self, hamiltonian, trial):
        """Compute mean field shift.

            .. math::

                \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

        """
        # hamiltonian.chol_vecs [X, M^2]
        if hamiltonian.sparse:
            mf_shift = 1j*hamiltonian.chol_vecs*trial.G[0].ravel()
            mf_shift += 1j*hamiltonian.chol_vecs*trial.G[1].ravel()
        else:
            Gcharge = (trial.G[0]+trial.G[1]).ravel()
            if numpy.isrealobj(hamiltonian.chol_vecs):
                tmp_real = numpy.dot(hamiltonian.chol_vecs.T, Gcharge.real)
                tmp_imag = numpy.dot(hamiltonian.chol_vecs.T, Gcharge.imag)
                mf_shift = 1.j * tmp_real - tmp_imag
            else:
                mf_shift = 1j*numpy.dot(hamiltonian.chol_vecs.T,
                                        (trial.G[0]+trial.G[1]).ravel())
        return mf_shift

    def construct_mean_field_shift_multi_det(self, hamiltonian, trial):
        if (trial.G != None):
            mf_shift = self.construct_mean_field_shift(hamiltonian,trial)
        else:
            nb = hamiltonian.nbasis
            mf_shift = [trial.contract_one_body(Vpq.reshape(nb,nb)) for Vpq in hamiltonian.chol_vecs.T]
            mf_shift = 1j*numpy.array(mf_shift)
        return mf_shift

    def construct_one_body_propagator(self, hamiltonian, dt):
        """Construct mean-field shifted one-body propagator.

        .. math::

            H1 \rightarrow H1 - v0
            v0_{ik} = \sum_n v_{(ik),n} \bar{v}_n

        Parameters
        ----------
        hamiltonian : hamiltonian class.
            Generic hamiltonian object.
        dt : float
            Timestep.
        """
        nb = hamiltonian.nbasis
        # shift = 1j*hamiltonian.chol_vecs.dot(self.mf_shift).reshape(nb,nb)
        shift = 1j*numpy.einsum("mx,x->m", hamiltonian.chol_vecs, self.mf_shift).reshape(nb,nb)
        H1 = hamiltonian.h1e_mod - numpy.array([shift,shift])
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_VHS_batch(self, hamiltonian, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        if is_cupy(hamiltonian.chol_vecs): # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy
            assert(cupy.is_available())
            isrealobj = cupy.isrealobj
        else:
            isrealobj = numpy.isrealobj

        if (hamiltonian.symmetry):
            if isrealobj(hamiltonian.chol_vecs):
                VHS_packed = hamiltonian.chol_packed.dot(xshifted.real) + 1.j * hamiltonian.chol_packed.dot(xshifted.imag)
            else:
                VHS_packed = hamiltonian.chol_packed.dot(xshifted)
            # (nb, nb, nw) -> (nw, nb, nb)
            VHS_packed = VHS_packed.T.reshape(self.nwalkers, hamiltonian.chol_packed.shape[0]).copy()
            VHS = numpy.zeros((self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis), dtype = VHS_packed.dtype)
            # for iw in range(self.nwalkers):
            unpack_VHS_batch(hamiltonian.sym_idx[0], hamiltonian.sym_idx[1], VHS_packed, VHS)
        else:
            if isrealobj(hamiltonian.chol_vecs):
                VHS = hamiltonian.chol_vecs.dot(xshifted.real) + 1.j * hamiltonian.chol_vecs.dot(xshifted.imag)
            else:
                VHS = hamiltonian.chol_vecs.dot(xshifted)
            # (nb, nb, nw) -> (nw, nb, nb)
            VHS = VHS.T.reshape(self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis).copy()

        return  self.isqrt_dt * VHS
