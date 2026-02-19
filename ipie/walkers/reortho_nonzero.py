from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import qr, qr_mode

def batched_qr_nonzero(A, mode=qr_mode):
    """
    Perform a batched QR where zero-columns remain zero:
      A : array of shape (w, a, b)
    Returns
      Q : array of shape (w, a, b) where, for each i in [0..w),
          Q[i,:,j] is the j-th column of the reduced-Q if A[i,:,j] was non-zero,
          or all zeros if A[i,:,j] was a zero-column.
    """
    w, _, _ = A.shape
    Q = xp.zeros_like(A)
    log_det_R = xp.zeros(w, dtype=A.dtype)

    for i in range(w):
        zero_cols = xp.all(A[i] == 0, axis=0)
        if zero_cols.all():
            continue
        Qi, Ri = qr(A[i][:, ~zero_cols], mode=mode)

        Ri_diag = xp.diag(Ri)
        signs_i = xp.sign(Ri_diag)
        Qi = xp.dot(Qi, xp.diag(signs_i))
        log_det_R[i] = xp.sum(xp.log(xp.abs(Ri_diag)))

        Q[i][:, ~zero_cols] = Qi

    return Q, log_det_R