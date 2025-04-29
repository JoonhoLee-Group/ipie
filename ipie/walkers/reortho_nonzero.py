from ipie.utils.backend import arraylib as xp

def batched_qr_nonzero(A, mode="reduced"):
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
        # 1) find zero columns in slice i
        zero_cols = xp.all(A[i] == 0, axis=0)               # shape (b,) :contentReference[oaicite:0]{index=0}

        # if every column is zero, skip
        if zero_cols.all():
            continue

        # 2) do reduced QR on only the nonzero columns
        #    A[i][:, ~zero_cols] has shape (a, k) with k ≤ b
        Qi, Ri = xp.linalg.qr(A[i][:, ~zero_cols], mode=mode)  # :contentReference[oaicite:1]{index=1}

        Ri_diag = xp.diag(Ri)
        signs_i = xp.sign(Ri_diag)
        Qi = xp.dot(Qi, xp.diag(signs_i))
        log_det_R[i] = xp.sum(xp.log(xp.abs(Ri_diag)))

        # 3) put those orthonormal columns back into Q
        Q[i][:, ~zero_cols] = Qi

    return Q, log_det_R