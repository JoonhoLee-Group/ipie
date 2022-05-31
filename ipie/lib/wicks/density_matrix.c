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
  /*printf("here\n");*/
  for (int idet = 0; idet < num_dets; idet++) {
    u_int64_t* det_bra = &dets[idet*DET_LEN];
    double ci_bra = ci_coeffs[idet];
    denom += ci_bra * ci_bra;
    /*printf("denom \n: %f", denom);*/
    fill_diagonal_term(
        det_bra,
        ci_bra,
        occs,
        density_matrix,
        num_orbs,
        nel
        );
    /*printf("\n");*/
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
          /*if (idet == 50 || jdet == 50)*/
            /*printf("%d %d %d %d %d %d %d %f %f\n", idet, jdet, ia[0], ia[1], spat_i, spat_a, perm, ci_ket, ci_bra);*/
          /*for (int i = 0; i < DET_LEN; i++) {*/
            /*printf("%d %llu\n", i, det_ket[i]);*/
          /*}*/
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
  /*printf("here\n");*/
  for (int idet = 0; idet < num_dets; idet++) {
    u_int64_t* det_bra = &dets[idet*DET_LEN];
    complex double ci_bra = ci_coeffs[idet];
    denom += conj(ci_bra) * ci_bra;
    /*printf("denom \n: %f", denom);*/
    fill_diagonal_term_cmplx(
        det_bra,
        ci_bra,
        occs,
        density_matrix,
        num_orbs,
        nel
        );
    /*printf("\n");*/
    for (int jdet = idet+1; jdet < num_dets; jdet++) {
      u_int64_t* det_ket = &dets[jdet*DET_LEN];
      complex double ci_ket = ci_coeffs[jdet];
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
          /*if (idet == 50 || jdet == 50)*/
            /*printf("%d %d %d %d %d %d %d %f %f\n", idet, jdet, ia[0], ia[1], spat_i, spat_a, perm, ci_ket, ci_bra);*/
          /*for (int i = 0; i < DET_LEN; i++) {*/
            /*printf("%d %llu\n", i, det_ket[i]);*/
          /*}*/
          density_matrix[pq] += perm * conj(ci_bra) * ci_ket;
          density_matrix[qp] += perm * ci_bra * conj(ci_ket);
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
  /*for (int i = 0; i < DET_LEN; i++)*/
    /*printf("D: %d\n", det[i]);*/
  decode_det(det, occs, nel);
  /*printf("nel : %d \n", nel);*/
  for (int iel = 0; iel < nel; iel++) {
    // density_matrix[spin,p,q]
    int spatial = occs[iel] / 2;
    int spin_offset = num_orbs*num_orbs*(occs[iel]%2);
    int pq = spatial*num_orbs + spatial + spin_offset;
    /*printf("%d %d %d \n", iel, spatial, occs[iel]%2);*/
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
  /*for (int i = 0; i < DET_LEN; i++)*/
    /*printf("D: %d\n", det[i]);*/
  decode_det(det, occs, nel);
  /*printf("nel : %d \n", nel);*/
  for (int iel = 0; iel < nel; iel++) {
    // density_matrix[spin,p,q]
    int spatial = occs[iel] / 2;
    int spin_offset = num_orbs*num_orbs*(occs[iel]%2);
    int pq = spatial*num_orbs + spatial + spin_offset;
    /*printf("%d %d %d \n", iel, spatial, occs[iel]%2);*/
    density_matrix[pq] += ci_coeff * ci_coeff;
  }
}
