#pragma once

#include <stdlib.h>

void encode_dets(
    const int *occsa,
    const int *occsb,
    u_int64_t *dets,
    const size_t nocca,
    const size_t noccb,
    const size_t ndet);
u_int64_t encode_det(
    const int *occa,
    const int *occb,
    const size_t nocca,
    const size_t noccb);
int count_set_bits(const u_int64_t det);
int get_excitation_level(
    const u_int64_t deta,
    const u_int64_t detb);
void decode_det(
    u_int64_t det,
    int* occs,
    const size_t nel);
void get_ia(
    u_int64_t det_bra,
    u_int64_t det_ket,
    int* ia);
int get_perm_ia(
    u_int64_t det,
    int i,
    int a);
