#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "determinant_utils.h"
#include "density_matrix.h"

void compute_density_matrix(
    double* ci_coeffs,
    u_int64_t* dets,
    double* density_matrix,
    int* occs,
    size_t num_dets,
    size_t num_orbs,
    size_t nel
    )
{
   // bras
  int ia[2];
  double denom = 0.0;
  for (int idet = 0; idet < num_dets; idet++) {
    u_int64_t* det_bra = &dets[idet*DET_LEN];
    double ci_bra = ci_coeffs[idet];
    denom += ci_bra * ci_bra;
    fill_diagonal_term(
        det_bra,
        ci_bra,
        occs,
        density_matrix,
        num_orbs,
        nel
        );
    for (int jdet = idet+1; jdet < num_dets; jdet++) {
      u_int64_t* det_ket = &dets[jdet*DET_LEN];
      double ci_ket = ci_coeffs[jdet];
      int excitation = get_excitation_level(det_bra, det_ket);
      if (excitation == 1) {
        get_ia(det_bra, det_ket, ia);
        int perm = get_perm_ia(det_ket, ia[0], ia[1]);
        int si = ia[0] % 2;
        int sa = ia[1] % 2;
        int spat_i = ia[0] / 2;
        int spat_a = ia[1] / 2;
        if (si == sa) {
          int spin_offset = num_orbs*num_orbs*si;
          int pq = spat_a*num_orbs + spat_i + spin_offset;
          int qp = spat_i*num_orbs + spat_a + spin_offset;
          density_matrix[pq] += perm * ci_bra * ci_ket;
          density_matrix[qp] += perm * ci_bra * ci_ket;
        }
      }
    }
  }
  for (int i = 0; i < num_orbs*num_orbs*2; i++) {
    density_matrix[i] = density_matrix[i] / denom;
  }
}

void compute_density_matrix_cmplx(
    complex double* ci_coeffs,
    u_int64_t* dets,
    complex double* density_matrix,
    int* occs,
    size_t num_dets,
    size_t num_orbs,
    size_t nel
    )
{
   // bras
  int ia[2];
  complex double denom = 0.0;
  for (int idet = 0; idet < num_dets; idet++) {
    u_int64_t* det_ket = &dets[idet*DET_LEN];
    complex double ci_ket = ci_coeffs[idet];
    denom += conj(ci_ket) * ci_ket;
    /*printf("denom \n: %f %f\n ", creal(denom), cimag(denom));*/
    fill_diagonal_term_cmplx(
        det_ket,
        ci_ket,
        occs,
        density_matrix,
        num_orbs,
        nel
        );
    for (int jdet = idet+1; jdet < num_dets; jdet++) {
      u_int64_t* det_bra = &dets[jdet*DET_LEN];
      complex double ci_bra = ci_coeffs[jdet];
      int excitation = get_excitation_level(det_bra, det_ket);
      if (excitation == 1) {
        get_ia(det_bra, det_ket, ia);
        int perm = get_perm_ia(det_ket, ia[0], ia[1]);
        int si = ia[0] % 2;
        int sa = ia[1] % 2;
        int spat_i = ia[0] / 2;
        int spat_a = ia[1] / 2;
        if (si == sa) {
          int spin_offset = num_orbs*num_orbs*si;
          int pq = spat_a*num_orbs + spat_i + spin_offset;
          int qp = spat_i*num_orbs + spat_a + spin_offset;
          complex double val = perm * conj(ci_bra) * ci_ket;
          density_matrix[pq] += val;
          density_matrix[qp] += conj(val);
        }
      }
    }
  }
  for (int i = 0; i < num_orbs*num_orbs*2; i++) {
    density_matrix[i] = density_matrix[i] / denom;
  }
}

void fill_diagonal_term(
    u_int64_t *det,
    double ci_coeff,
    int* occs,
    double* density_matrix,
    size_t num_orbs,
    size_t nel
    )
{
  decode_det(det, occs, nel);
  for (int iel = 0; iel < nel; iel++) {
    int spatial = occs[iel] / 2;
    int spin_offset = num_orbs*num_orbs*(occs[iel]%2);
    int pq = spatial*num_orbs + spatial + spin_offset;
    density_matrix[pq] += ci_coeff * ci_coeff;
  }
}

void fill_diagonal_term_cmplx(
    u_int64_t *det,
    complex double ci_coeff,
    int* occs,
    complex double* density_matrix,
    size_t num_orbs,
    size_t nel
    )
{
  decode_det(det, occs, nel);
  for (int iel = 0; iel < nel; iel++) {
    int spatial = occs[iel] / 2;
    int spin_offset = num_orbs*num_orbs*(occs[iel]%2);
    int pq = spatial*num_orbs + spatial + spin_offset;
    density_matrix[pq] += conj(ci_coeff) * ci_coeff;
  }
}
