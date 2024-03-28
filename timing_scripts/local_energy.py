import os

os.environ["MKL_NUM_THREADS"] = "1"
import timeit

import jax
import jax.numpy as jnp
import numpy as np


def numpy_compute(rchol_a, rchol_b, GaT, GbT):
    naux = rchol_a.shape[0]
    exx = 0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        Ta = rchol_a[x].dot(GaT)  # this is a (nalpha, nalpha)
        exx += np.trace(Ta.dot(Ta))
        Tb = rchol_b[x].dot(GbT)  # this is (nbeta, nbeta)
        exx += np.trace(Tb.dot(Tb))
    return exx


@jax.jit
def jax_chol(rchol_a_x, GaT):
    Ta = rchol_a_x.dot(GaT)  # this is a (nalpha, nalpha)
    return jnp.trace(Ta.dot(Ta))


@jax.jit
def jax_compute(rchol_a, rchol_b, GaT, GbT):
    vmap_jax_cc = jax.vmap(jax_chol, in_axes=[0, None])
    exx_a = jnp.sum(vmap_jax_cc(rchol_a, GaT))
    exx_b = jnp.sum(vmap_jax_cc(rchol_b, GbT))
    return exx_a + exx_b


def load_func(as_jax=False):
    # GaT = np.load("GaT.npy")
    # GbT = np.load("GbT.npy")
    # rchol_a = np.load("rchol_a.npy")
    # rchol_b = np.load("rchol_b.npy")
    nao = 570
    nocc = 120
    naux = 3000
    GaT = np.zeros((nao, nocc), dtype=np.complex128)
    GbT = np.zeros((nao, nocc), dtype=np.complex128)
    rchol_a = np.zeros((naux, nocc, nao), dtype=np.float64)
    rchol_b = np.zeros((naux, nocc, nao), dtype=np.float64)
    if as_jax:
        return jnp.array(rchol_a), jnp.array(rchol_b), jnp.array(GaT), jnp.array(GbT)
    return rchol_a, rchol_b, GaT, GbT


def main():
    print([device.device_kind for device in jax.local_devices()])
    print(np.show_config())
    nao = 570
    nocc = 120
    naux = 3000
    GaT = np.zeros((nao, nocc), dtype=np.complex128)
    GbT = np.zeros((nao, nocc), dtype=np.complex128)
    rchol_a = np.zeros((naux, nocc, nao), dtype=np.float64)
    rchol_b = np.zeros((naux, nocc, nao), dtype=np.float64)
    # GaT = np.load("GaT.npy")
    # GbT = np.load("GbT.npy")
    # rchol_a = np.load("rchol_a.npy")
    # rchol_b = np.load("rchol_b.npy")

    vmap_jax_cc = jax.vmap(jax_chol, in_axes=[0, None])
    exx_a = np.sum(vmap_jax_cc(jnp.array(rchol_a), jnp.array(GaT)))
    exx_b = np.sum(vmap_jax_cc(jnp.array(rchol_b), jnp.array(GbT)))
    exx = exx_a + exx_b
    val = jax_compute(
        jnp.array(rchol_a), jnp.array(rchol_b), jnp.array(GaT), jnp.array(GbT)
    ).block_until_ready()
    print(val, exx)
    t = timeit.timeit(
        stmt="jax_compute(rchol_a, rchol_b, GaT, GbT).block_until_ready()",
        setup="rchol_a, rchol_b, GaT, GbT = load_func(as_jax=True)",
        globals=globals(),
        number=1,
    )
    print(t)

    t = timeit.timeit(
        stmt="numpy_compute(rchol_a, rchol_b, GaT, GbT)",
        setup="rchol_a, rchol_b, GaT, GbT = load_func()",
        globals=globals(),
        number=1,
    )
    print(t)


if __name__ == "__main__":
    main()
