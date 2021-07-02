import h5py
import numpy
try:
    from mpi4py import MPI
    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg
import time
from pauxy.estimators.utils import H5EstimatorHelper
from pauxy.estimators.ci import get_hmatel
from pauxy.estimators.thermal import particle_number, one_rdm_from_G
try:
    from pauxy.estimators.ueg import local_energy_ueg
    from pauxy.estimators.pw_fft import local_energy_pw_fft
except ImportError as e:
    print(e)
from pauxy.estimators.hubbard import local_energy_hubbard, local_energy_hubbard_ghf,\
                                     local_energy_hubbard_holstein
from pauxy.estimators.greens_function import gab_mod_ovlp, gab_mod
from pauxy.estimators.generic import (
    local_energy_generic_opt,
    local_energy_generic,
    local_energy_generic_pno,
    local_energy_generic_cholesky,
    local_energy_generic_cholesky_opt,
    local_energy_generic_cholesky_opt_stochastic
)
from pauxy.utils.io import format_fixed_width_strings, format_fixed_width_floats
from pauxy.utils.misc import dotdict


class Mixed(object):
    """Class for computing mixed estimates.

    Parameters
    ----------
    mixed : dict
        Input options for mixed estimates.
    root : bool
        True if on root/master processor.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.

    Attributes
    ----------
    nmeasure : int
        Max number of measurements.
    nreg : int
        Number of regular estimates (exluding iteration).
    G : :class:`numpy.ndarray`
        One-particle RDM.
    estimates : :class:`numpy.ndarray`
        Store for mixed estimates per processor.
    global_estimates : :class:`numpy.ndarray`
        Store for mixed estimates accross all processors.
    names : :class:`pauxy.estimators.EstimEnum`
        Enum for locating estimates in estimates array.
    header : int
        Output header.
    key : dict
        Explanation of output.
    output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting data to HDF5 group.
    output : :class:`pauxy.estimators.H5EstimatorHelper`
        Class for outputting rdm data to HDF5 group.
    """

    def __init__(self, mixed, system, root, filename, qmc, trial, dtype):
        self.average_gf = mixed.get('average_gf', False)
        self.eval_energy = mixed.get('evaluate_energy', True)
        self.calc_one_rdm = mixed.get('one_rdm', False)
        self.calc_two_rdm = mixed.get('two_rdm', None)
        self.energy_eval_freq = mixed.get('energy_eval_freq', None)
        if self.energy_eval_freq is None:
            self.energy_eval_freq = qmc.nsteps
        self.verbose = mixed.get('verbose', True)
        # number of steps per block
        self.nsteps = qmc.nsteps
        self.header = ['Iteration', 'WeightFactor', 'Weight', 'ENumer',
                       'EDenom', 'ETotal', 'E1Body', 'E2Body', 'EHybrid',
                       'Overlap']
        if qmc.beta is not None:
            self.thermal = True
            self.header.append('Nav')
        else:
            self.thermal = False
        self.header.append('Time')
        self.nreg = len(self.header[1:])
        self.dtype = dtype
        self.G = numpy.zeros((2,system.nbasis,system.nbasis), dtype)
        if self.calc_one_rdm:
            dms_size = self.G.size
        else:
            dms_size = 0
        self.eshift = numpy.array([0,0])
        # Abuse of language for the moment. Only accumulates S(k) for UEG.
        # TODO: Add functionality to accumulate 2RDM?
        if self.calc_two_rdm is not None:
            if self.calc_two_rdm == "structure_factor":
                two_rdm_shape = (2,2,len(system.qvecs),)
            self.two_rdm = numpy.zeros(two_rdm_shape,
                                       dtype=numpy.complex128)
            dms_size += self.two_rdm.size
        else:
            self.two_rdm = None
        self.estimates = numpy.zeros(self.nreg+dms_size, dtype=dtype)
        self.names = get_estimator_enum(self.thermal)
        self.estimates[self.names.time] = time.time()
        self.global_estimates = numpy.zeros(self.nreg+dms_size,
                                            dtype=dtype)
        self.key = {
            'Iteration': "Simulation iteration. iteration*dt = tau.",
            'WeightFactor': "Rescaling Factor from population control.",
            'Weight': "Total walker weight.",
            'E_num': "Numerator for projected energy estimator.",
            'E_denom': "Denominator for projected energy estimator.",
            'ETotal': "Projected energy estimator.",
            'E1Body': "Mixed one-body energy estimator.",
            'E2Body': "Mixed two-body energy estimator.",
            'EHybrid': "Hybrid energy.",
            'Overlap': "Walker average overlap.",
            'Nav': "Average number of electrons.",
            'Time': "Time per processor to complete one iteration.",
        }
        if root:
            self.setup_output(filename)

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update mixed estimates for walkers.

        Parameters
        ----------
        system : system object.
            Container for model input options.
        qmc : :class:`pauxy.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`pauxy.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`pauxy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        if free_projection:
            for i, w in enumerate(psi.walkers):
                # For T > 0 w.ot = 1 always.
                wfac = w.weight * w.ot * w.phase# * numpy.exp(w.log_detR-w.log_detR_shift)
                if step % self.energy_eval_freq == 0:
                    w.greens_function(trial)
                    if self.eval_energy:
                        if self.thermal:
                            E, T, V = w.local_energy(system)
                        else:
                            E, T, V = w.local_energy(system, rchol=trial._rchol, eri=trial._eri)
                    else:
                        E, T, V = 0, 0, 0
                    self.estimates[self.names.enumer] += wfac * E
                    self.estimates[self.names.e1b:self.names.e2b+1] += (
                            wfac * numpy.array([T,V])
                    )
                    self.estimates[self.names.edenom] += wfac
                if self.thermal:
                    nav = particle_number(one_rdm_from_G(w.G))
                    self.estimates[self.names.nav] += wfac * nav
                self.estimates[self.names.uweight] += w.unscaled_weight
                self.estimates[self.names.weight] += wfac
                self.estimates[self.names.ehyb] += wfac * w.hybrid_energy
                self.estimates[self.names.ovlp] += w.weight * abs(w.ot)
        else:
            # When using importance sampling we only need to know the current
            # walkers weight as well as the local energy, the walker's overlap
            # with the trial wavefunction is not needed.
            for i, w in enumerate(psi.walkers):
                if self.thermal:
                    if self.average_gf:
                        E_sum = 0
                        T_sum = 0
                        V_sum = 0
                        nav = 0
                        for ts in range(w.stack_length):
                            w.greens_function(trial, slice_ix=ts*w.stack_size)
                            E, T, V = w.local_energy(system,
                                                     two_rdm=self.two_rdm)
                            E_sum += E
                            T_sum += T
                            V_sum += V
                            nav += particle_number(one_rdm_from_G(w.G))
                        self.estimates[self.names.nav] += w.weight * nav / w.stack_length
                        self.estimates[self.names.enumer] += w.weight*E_sum.real/w.stack_length
                        self.estimates[self.names.e1b:self.names.e2b+1] += (
                                w.weight*numpy.array([T_sum,V_sum]).real/w.stack_length
                        )
                    else:
                        w.greens_function(trial)
                        E, T, V = w.local_energy(system, two_rdm=self.two_rdm)
                        nav = particle_number(one_rdm_from_G(w.G))
                        self.estimates[self.names.nav] += w.weight * nav
                        self.estimates[self.names.enumer] += w.weight*E.real
                        self.estimates[self.names.e1b:self.names.e2b+1] += (
                                w.weight*numpy.array([T,V]).real
                        )
                        self.estimates[self.names.edenom] += w.weight
                else:
                    if step % self.energy_eval_freq == 0:
                        w.greens_function(trial)
                        if self.eval_energy:
                            E, T, V = w.local_energy(system, rchol=trial._rchol, eri=trial._eri, UVT=trial._UVT)
                        else:
                            E, T, V = 0, 0, 0
                        self.estimates[self.names.enumer] += w.weight*w.le_oratio*E.real
                        self.estimates[self.names.e1b:self.names.e2b+1] += (
                                w.weight*w.le_oratio*numpy.array([T,V]).real
                        )
                        self.estimates[self.names.edenom] += w.weight * w.le_oratio
                self.estimates[self.names.uweight] += w.unscaled_weight
                self.estimates[self.names.weight] += w.weight
                self.estimates[self.names.ovlp] += w.weight * abs(w.ot)
                self.estimates[self.names.ehyb] += w.weight * w.hybrid_energy
                if self.calc_one_rdm:
                    start = self.names.time+1
                    end = self.names.time+1+w.G.size
                    self.estimates[start:end] += w.weight*w.G.flatten().real
                if self.calc_two_rdm is not None:
                    start = end
                    end = end + self.two_rdm.size
                    self.estimates[start:end] += w.weight*self.two_rdm.flatten().real

    def print_step(self, comm, nprocs, step, nsteps=None, free_projection=False):
        """Print mixed estimates to file.

        This reduces estimates arrays over processors. On return estimates
        arrays are zerod.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        if step % self.nsteps != 0:
            return
        if nsteps is None:
            nsteps = self.nsteps
        es = self.estimates
        ns = self.names
        es[ns.time] = (time.time()-es[ns.time]) / nprocs
        es[ns.uweight:ns.weight+1] /= nsteps
        es[ns.ehyb:ns.time+1] /= nsteps
        comm.Reduce(es, self.global_estimates, op=mpi_sum)
        gs = self.global_estimates
        if comm.rank == 0:
            gs[ns.eproj] = gs[ns.enumer]
            gs[ns.eproj:ns.e2b+1] = gs[ns.eproj:ns.e2b+1] / gs[ns.edenom]
            gs[ns.ehyb] /= gs[ns.weight]
            gs[ns.ovlp] /= gs[ns.weight]
            eshift = numpy.array([gs[ns.ehyb],gs[ns.eproj]])
        else:
            eshift = numpy.array([0,0])
        if self.thermal and comm.rank == 0:
            gs[ns.nav] = gs[ns.nav] / gs[ns.weight]
        eshift = comm.bcast(eshift, root=0)
        self.eshift = eshift
        if comm.rank == 0:
            if self.verbose:
                print(format_fixed_width_floats([step]+list(gs[:ns.time+1].real)))
            self.output.push([step]+list(gs[:ns.time+1]), 'energies')
            if self.calc_one_rdm:
                start = self.nreg
                end = self.nreg+self.G.size
                rdm = gs[start:end].reshape(self.G.shape) / nsteps
                self.output.push(rdm/gs[ns.weight], 'one_rdm')
            if self.calc_two_rdm:
                start = self.nreg + self.G.size
                rdm = gs[start:].reshape(self.two_rdm.shape) / nsteps
                self.output.push(rdm/gs[ns.weight], 'two_rdm')
            self.output.increment()
        self.zero()

    def print_key(self, eol='', encode=False):
        """Print out information about what the estimates are.

        Parameters
        ----------
        eol : string, optional
            String to append to output, e.g., Default : ''.
        encode : bool
            In True encode output to be utf-8.
        """
        header = (
            eol + '# Explanation of output column headers:\n' +
            '# -------------------------------------' + eol
        )
        if encode:
            header = header.encode('utf-8')
        print(header)
        for (k, v) in self.key.items():
            s = '# %s : %s' % (k, v) + eol
            if encode:
                s = s.encode('utf-8')
            print(s)

    def print_header(self, eol='', encode=False):
        r"""Print out header for estimators

        Parameters
        ----------
        eol : string, optional
            String to append to output, Default : ''.
        encode : bool
            In True encode output to be utf-8.

        Returns
        -------
        None
        """
        s = format_fixed_width_strings(self.header) + eol
        if encode:
            s = s.encode('utf-8')
        print(s)

    def projected_energy(self):
        """Computes projected energy from estimator array.

        Returns
        -------
        eproj : float
            Mixed estimate for projected energy.
        """
        numerator = self.estimates[self.names.enumer]
        denominator = self.estimates[self.names.edenom]
        return (numerator / denominator).real

    def get_shift(self, hybrid=True):
        """get hybrid shift.

        parameters
        ----------
        hybrid : bool
            true if using hybrid propgation
        returns
        -------
        eshift : float
            walker averaged hybrid energy.
        """
        if hybrid:
            return self.eshift[0].real
        else:
            return self.eshift[1].real

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0
        self.estimates[self.names.time] = time.time()

    def setup_output(self, filename):
        with h5py.File(filename, 'a') as fh5:
            fh5['basic/headers'] = numpy.array(self.header).astype('S')
        self.output = H5EstimatorHelper(filename, 'basic')

# Energy evaluation routines for the Hubbard-Holstein model.
def local_energy_hh(system, G, X, Lap, Ghalf=None):
    if system.name == "HubbardHolstein":
        (e1, e2, e3) = local_energy_hubbard_holstein(system, G, X, Lap, Ghalf)
        return (e1, e2, e3)
    else:
        print("SOMETHING IS VERY WRONG... WHY ARE YOU CALLING HUBBARD-HOSTEIN FUNCTION?")
        exit()

# Energy evaluation routines.
def local_energy(system, G, Ghalf=None,
                 two_rdm=None,
                 rchol=None, eri=None, C0=None, ecoul0=None, exxa0=None, exxb0=None, UVT=None):
    """Helper routine to compute local energy.

    Parameters
    ----------
    system : system object
        system object.
    G : :class:`numpy.ndarray`
        1RDM.
    C0 : :class:`numpy.ndarray`
        trial C.

    Returns
    -------
    (E,T,V) : tuple
        Total, one-body and two-body energy.
    """
    ghf = (G.shape[-1] == 2*system.nbasis)
    # unfortunate interfacial problem for the HH model
    if system.name == "Hubbard" or system.name == "HubbardHolstein":
        if ghf:
            return local_energy_ghf(system, G)
        else:
            return local_energy_hubbard(system, G)
    elif system.name == "PW_FFT":
            return local_energy_pw_fft(system, G, Ghalf, two_rdm=two_rdm)
    elif system.name == "UEG":
        return local_energy_ueg(system, G, two_rdm=two_rdm)
    else:
        if Ghalf is not None:
            if system.stochastic_ri and system.control_variate:
                return local_energy_generic_cholesky_opt_stochastic(system, G,
                                     nsamples=system.nsamples,
                                     Ghalf=Ghalf,
                                     rchol=rchol, C0=C0, ecoul0=ecoul0,
                                     exxa0=exxa0,
                                     exxb0=exxb0)
            elif system.stochastic_ri and not system.control_variate:
                return local_energy_generic_cholesky_opt_stochastic(system, G,
                                     nsamples=system.nsamples,
                                     Ghalf=Ghalf,
                                     rchol=rchol)
            elif system.exact_eri and not system.pno:
                return local_energy_generic_opt(system, G, Ghalf=Ghalf, eri=eri)
            elif system.pno:
                assert(system.exact_eri and system.control_variate)
                return local_energy_generic_pno(system, G, Ghalf=Ghalf, eri=eri, C0=C0, ecoul0=ecoul0, exxa0=exxa0, exxb0=exxb0, UVT=UVT)
            else:
                return local_energy_generic_cholesky_opt(system, G,
                                                         Ghalf=Ghalf,
                                                         rchol=rchol)
        else:
            return local_energy_generic_cholesky(system, G)

def local_energy_multi_det(system, Gi, weights, two_rdm=None, rchol=None):
    weight = 0
    energies = 0
    denom = 0
    for w, G in zip(weights, Gi):
        # construct "local" green's functions for each component of A
        energies += w * numpy.array(local_energy(system, G, rchol = rchol))
        denom += w

    return tuple(energies/denom)

def local_energy_multi_det_hh(system, Gi, weights, X, Lapi, two_rdm=None):
    weight = 0
    energies = 0
    denom = 0
    for w, G, Lap in zip(weights, Gi, Lapi):
        # construct "local" green's functions for each component of A
        energies += w * numpy.array(local_energy_hubbard_holstein(system, G, X, Lap, Ghalf=None))
        denom += w
    return tuple(energies/denom)

def get_estimator_enum(thermal=False):
    keys = ['uweight', 'weight', 'enumer', 'edenom',
            'eproj', 'e1b', 'e2b', 'ehyb', 'ovlp']
    if thermal:
        keys.append('nav')
    keys.append('time')
    enum = {}
    for v, k in enumerate(keys):
        enum[k] = v
    return dotdict(enum)


def eproj(estimates, enum):
    """Real projected energy.

    Parameters
    ----------
    estimates : numpy.array
        Array containing estimates averaged over all processors.
    enum : :class:`pauxy.estimators.EstimatorEnum` object
        Enumerator class outlining indices of estimates array elements.

    Returns
    -------
    eproj : float
        Projected energy from current estimates array.
    """

    numerator = estimates[enum.enumer]
    denominator = estimates[enum.edenom]
    return (numerator/denominator).real

def variational_energy(system, psi, coeffs, G=None, GH=None, rchol=None, eri=None, 
                       C0 = None,ecoul0 =None,exxa0 = None,exxb0 = None,UVT=None):
    if len(psi.shape) == 2:
        return variational_energy_single_det(system, psi,
                                             G=G, GH=GH,
                                             rchol=rchol, eri=eri, 
                                             C0 = C0, ecoul0 = ecoul0, 
                                             exxa0 = exxa0, exxb0 = exxb0,
                                             UVT=UVT)
    elif len(psi) == 1:
        return variational_energy_single_det(system, psi[0],
                                             G=G, GH=GH,
                                             rchol=rchol, eri=eri, 
                                             C0 = C0, ecoul0 = ecoul0, 
                                             exxa0 = exxa0, exxb0 = exxb0,
                                             UVT=UVT)
    else:
        return variational_energy_multi_det(system, psi, coeffs)

def variational_energy_multi_det(system, psi, coeffs, H=None, S=None):
    weight = 0
    energies = 0
    denom = 0
    nup = system.nup
    ndet = len(coeffs)
    if H is not None and S is not None:
        store = True
    else:
        store = False
    for i, (Bi, ci) in enumerate(zip(psi, coeffs)):
        for j, (Aj, cj) in enumerate(zip(psi, coeffs)):
            # construct "local" green's functions for each component of A
            Gup, GHup, inv_O_up = gab_mod_ovlp(Bi[:,:nup], Aj[:,:nup])
            Gdn, GHdn, inv_O_dn = gab_mod_ovlp(Bi[:,nup:], Aj[:,nup:])
            ovlp = 1.0 / (scipy.linalg.det(inv_O_up)*scipy.linalg.det(inv_O_dn))
            weight = (ci.conj()*cj) * ovlp
            G = numpy.array([Gup, Gdn])
            e = numpy.array(local_energy(system, G))
            if store:
                H[i,j] = ovlp*e[0]
                S[i,j] = ovlp
            energies += weight * e
            denom += weight
    return tuple(energies/denom)

def variational_energy_ortho_det(system, occs, coeffs):
    """Compute variational energy for CI-like multi-determinant expansion.

    Parameters
    ----------
    system : :class:`pauxy.system` object
        System object.
    occs : list of lists
        list of determinants.
    coeffs : :class:`numpy.ndarray`
        Expansion coefficients.

    Returns
    -------
    energy : tuple of float / complex
        Total energies: (etot,e1b,e2b).
    """
    evar = 0.0
    denom = 0.0
    one_body = 0.0
    two_body = 0.0
    for i, (occi, ci) in enumerate(zip(occs, coeffs)):
        denom += ci.conj()*ci
        for j in range(0,i+1):
            cj = coeffs[j]
            occj = occs[j]
            etot, e1b, e2b = ci.conj()*cj*get_hmatel(system, occi, occj)
            evar += etot
            one_body += e1b
            two_body += e2b
            if j < i:
                # Use Hermiticity
                evar += etot
                one_body += e1b
                two_body += e2b
    return evar/denom, one_body/denom, two_body/denom

def variational_energy_single_det(system, psi, G=None, GH=None, 
    rchol=None, eri=None,
    C0=None, 
    ecoul0=None,
    exxa0=None,
    exxb0=None,  UVT=None):
    assert len(psi.shape) == 2
    return local_energy(system, G, Ghalf=GH, rchol=rchol, eri=eri, C0=C0, ecoul0=ecoul0, exxa0=exxa0, exxb0=exxb0, UVT=UVT)
