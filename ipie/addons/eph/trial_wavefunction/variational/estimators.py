import numpy as np
import jax.numpy as npj
from jax.config import config
config.update("jax_enable_x64", True)

def gab(A,B):
    inv_O = npj.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB
