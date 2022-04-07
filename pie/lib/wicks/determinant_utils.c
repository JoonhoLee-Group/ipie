#include <stdlib.h>
#include <stdio.h>
#include "determinant_utils.h"

void encode_dets(
    const int *occsa,
    const int *occsb,
    u_int64_t *dets,
    const size_t nocca,
    const size_t noccb,
    const size_t ndet)
{
  for (int idet = 0; idet < ndet; idet++) {
    dets[idet] = encode_det(
               &(occsa[idet*nocca]),
               &(occsb[idet*noccb]),
               nocca,
               noccb
               );
  }
}

u_int64_t encode_det(
    const int *occa,
    const int *occb,
    const size_t nocca,
    const size_t noccb)
{
  u_int64_t det = 0;
  u_int64_t mask = 1;
  for (int i = 0; i < nocca; i++) {
    /*printf("%d %d\n", occa[i], nocca);*/
    det |= (mask << 2*occa[i]);
  }
  for (int i = 0; i < noccb; i++) {
    det |= (mask << (2*occb[i] + 1));
  }
  /*printf("end\n");*/
  return det;
}

int count_set_bits(const u_int64_t deta)
{
  return __builtin_popcountll(deta);
}

int get_excitation_level(
    const u_int64_t deta,
    const u_int64_t detb)
{
  return count_set_bits(deta^detb) / 2;
}

void decode_det(
    const u_int64_t det,
    int* occs,
    const size_t nel
    )
{
  // dumb for the moment.
  int nset = 0;
  u_int64_t pos = 0;
  u_int64_t mask = 1;
  for (int i=0; i<64; i++) {
    if (det & (mask<<pos)) {
      occs[nset] = pos;
      nset++;
    }
    if (nset == nel)
      break;
    pos++;
  }
}

// < bra | a^ i | ket >
void get_ia(
    u_int64_t det_bra,
    u_int64_t det_ket,
    int* ia)
{
  u_int64_t delta = det_bra ^ det_ket;
  int diff[2];
  decode_det(delta, diff, 2);
  u_int64_t loc = 1;
  int a = 0;
  if (det_ket & (loc << diff[0])) {
    ia[0] = diff[0];
    ia[1] = diff[1];
  } else {
    ia[0] = diff[1];
    ia[1] = diff[0];
  }
}

// phase( a^i|ket> )
int get_perm_ia(
    u_int64_t det_ket,
    int i,
    int a)
{
  u_int64_t loc = 1;
  u_int64_t mask;
  // check bit a is occupied or bit i is unoccupied.
  // else just count set bits between i and a.
  if (a == i) {
    return 1;
  } else {
    if ((det_ket & (loc << a)) || !(det_ket & (loc << i))) {
      return 0;
    } else if (i > a) {
      mask = (loc << i) - (loc << a);
    } else {
      mask = (loc << a) - (loc << i);
    }
    /*printf("this: %d %d %d\n", mask, det_ket, count_set_bits(det_ket & mask));*/
    if (count_set_bits(det_ket & mask)%2 == 0) {
      return 1;
    } else {
      return -1;
    }
  }
}
