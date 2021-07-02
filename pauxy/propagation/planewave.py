import cmath
import math
import numpy
import scipy.sparse.linalg
import time
import sys
from pauxy.propagation.operations import local_energy_bound
from pauxy.utils.linalg import exponentiate_matrix, reortho
from pauxy.walkers.single_det import SingleDetWalker

class PlaneWave(object):
    """PlaneWave class
    """
    def __init__(self, system, trial, qmc, options={}, verbose=False):
        if verbose:
            print ("# Parsing plane wave propagator input options.")
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j*self.sqrt_dt
        self.mf_core = 0
        self.num_vplus = system.nfields // 2
        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)
        # Mean-field shift is zero for UEG.
        self.mf_shift = numpy.zeros(system.nfields, dtype=numpy.complex128)
        optimised = options.get('optimised', True)
        if optimised:
            self.construct_force_bias = self.construct_force_bias_incore
            self.construct_VHS = self.construct_VHS_incore
        else:
            print("# Slow routines not available. Please Implement.")
            sys.exit()
            # self.construct_force_bias = self.construct_force_bias_slow
            # self.construct_VHS = self.construct_VHS_slow
        # Input options
        if verbose:
            print ("# Finished setting up plane wave propagator.")

    def construct_one_body_propagator(self, system, dt):
        """Construct the one-body propagator Exp(-dt/2 H0)
        Parameters
        ----------
        system :
            system class
        dt : float
            time-step
        Returns
        -------
        self.BH1 : numpy array
            Exp(-dt/2 H0)
        """
        H1 = system.h1e_mod
        # No spin dependence for the moment.
        self.BH1 = numpy.array([scipy.linalg.expm(-0.5*dt*H1[0]),
                                scipy.linalg.expm(-0.5*dt*H1[1])])

    def construct_force_bias_incore(self, system, walker, trial):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        system :
            system class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        G = walker.G
        Gvec = G.reshape(2, system.nbasis*system.nbasis)
        self.vbias[:self.num_vplus] = Gvec[0].T*system.iA + Gvec[1].T*system.iA
        self.vbias[self.num_vplus:] = Gvec[0].T*system.iB + Gvec[1].T*system.iB
        # print(-self.sqrt_dt*self.vbias)
        # sys.exit()
        return - self.sqrt_dt * self.vbias

    def construct_VHS_incore(self, system, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        system :
            system class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        return construct_VHS_incore(system, xshifted, self.sqrt_dt)


def construct_VHS_incore(system, xshifted, sqrt_dt):
    """Construct the one body potential from the HS transformation
    Parameters
    ----------
    system :
        system class
    xshifted : numpy array
        shifited auxiliary field
    Returns
    -------
    VHS : numpy array
        the HS potential
    """
    VHS = numpy.zeros((system.nbasis, system.nbasis),
                      dtype=numpy.complex128)
    VHS = (system.iA * xshifted[:system.nchol] +
           system.iB * xshifted[system.nchol:])
    VHS = VHS.reshape(system.nbasis, system.nbasis)
    return  sqrt_dt * VHS

def construct_propagator_matrix_planewave(system, BT2, config, dt):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with generic system object.

    Parameters
    ----------
    system : class
        System class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    config : numpy array
        Auxiliary field configuration.
    conjt : bool
        If true return Hermitian conjugate of matrix.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full propagator matrix.
    """
    VHS = construct_VHS_incore(system, config, dt**0.5)
    EXP_VHS = exponentiate_matrix(VHS)
    Bup = BT2[0].dot(EXP_VHS).dot(BT2[0])
    Bdown = BT2[1].dot(EXP_VHS).dot(BT2[1])
    return numpy.array([Bup, Bdown])

def back_propagate_planewave(phi, stack, system, nstblz, BT2, dt, store=False):
    r"""Perform back propagation for RHF/UHF style wavefunction.

    For use with generic system hamiltonian.

    Parameters
    ---------
    system : system object in general.
        Container for model input options.
    psi : :class:`pauxy.walkers.Walkers` object
        CPMC wavefunction.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    dt : float
        Timestep.

    Returns
    -------
    psi_bp : list of :class:`pauxy.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(stack.get_block()[0][::-1]):
        B = construct_propagator_matrix_planewave(system, BT2, c, dt)
        phi[:,:nup] = numpy.dot(B[0].conj().T, phi[:,:nup])
        phi[:,nup:] = numpy.dot(B[1].conj().T, phi[:,nup:])
        if i != 0 and i % nstblz == 0:
            (phi[:,:nup], R) = reortho(phi[:,:nup])
            (phi[:,nup:], R) = reortho(phi[:,nup:])
        if store:
            psi_store.append(phi.copy())

    return psi_store

def unit_test():
    from pauxy.systems.ueg import UEG
    from pauxy.qmc.options import QMCOpts
    from pauxy.trial_wavefunction.hartree_fock import HartreeFock
    from pauxy.propagation.continuous import Continuous

    inputs = {'nup':1, 'ndown':1,
    'rs':1.0, 'ecut':1.0, 'dt':0.05, 'nwalkers':10}

    system = UEG(inputs, True)

    qmc = QMCOpts(inputs, system, True)

    trial = HartreeFock(system, False, inputs, True)

    propagator = Continuous(system, trial, qmc, verbose=True)


if __name__=="__main__":
    unit_test()
