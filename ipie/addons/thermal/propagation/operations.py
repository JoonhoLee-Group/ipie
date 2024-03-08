from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.misc import is_cupy

def apply_exponential(VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation

    Parameters
    ----------
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential

    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    phi = xp.identity(VHS.shape[-1], dtype=xp.complex128)
    Temp = xp.zeros(phi.shape, dtype=phi.dtype)
    xp.copyto(Temp, phi)

    for n in range(1, exp_nmax + 1):
        Temp = VHS.dot(Temp) / n
        phi += Temp

    return phi # Shape (nbasis, nbasis).
