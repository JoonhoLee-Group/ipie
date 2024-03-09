#pragma once

#include <stdlib.h>

#define DET_LEN 4
#define DET_SIZE 64

void encode_dets(const int *occsa, const int *occsb, u_int64_t *dets,
                 const size_t nocca, const size_t noccb, const size_t ndet);
void encode_det(const int *occa, const int *occb, u_int64_t *det,
                const size_t nocca, const size_t noccb);
int count_set_bits(const u_int64_t *det);
int count_set_bits_single(const u_int64_t det);
int get_excitation_level(const u_int64_t *deta, const u_int64_t *detb);
void decode_det(const u_int64_t *det, int *occs, const size_t nel);
void get_ia(u_int64_t *det_bra, u_int64_t *det_ket, int *ia);
int get_perm_ia(u_int64_t *det, int i, int a);
void bitwise_subtract(u_int64_t *deta, u_int64_t *detb, u_int64_t *result);
void bitwise_and(u_int64_t *deta, u_int64_t *detb, u_int64_t *result);
void build_set_mask(u_int64_t *mask, int det_loc, int det_ind);
