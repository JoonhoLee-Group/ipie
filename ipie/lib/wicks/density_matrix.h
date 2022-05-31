#pragma once

#include <complex.h>

void compute_density_matrix(
    double* ci_coeffs,
    u_int64_t* dets,
    double* density_matrix,
    int* occs,
    size_t num_dets,
    size_t num_orbs,
    size_t nel
    );

void compute_density_matrix_cmplx(
    complex double* ci_coeffs,
    u_int64_t* dets,
    complex double* density_matrix,
    int* occs,
    size_t num_dets,
    size_t num_orbs,
    size_t nel
    );

void fill_diagonal_term(
    u_int64_t *det,
    double ci_coeff,
    int* occs,
    double* density_matrix,
    size_t num_orbs,
    size_t nel
    );

void fill_diagonal_term_cmplx(
    u_int64_t *det,
    complex double ci_coeff,
    int* occs,
    complex double* density_matrix,
    size_t num_orbs,
    size_t nel
    );
