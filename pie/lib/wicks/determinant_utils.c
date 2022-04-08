#include <stdlib.h>
#include <stdio.h>
#include "determinant_utils.h"

// not planning on > 64 spatial orbs in active space.

void encode_dets(
    const int *occsa,
    const int *occsb,
    u_int64_t *dets,
    const size_t nocca,
    const size_t noccb,
    const size_t ndet)
{
  for (int idet = 0; idet < ndet; idet++) {
    encode_det(
               &(occsa[idet*nocca]),
               &(occsb[idet*noccb]),
               &(dets[idet*DET_LEN]),
               nocca,
               noccb
               );
  }
}

void encode_det(
    const int *occa,
    const int *occb,
    u_int64_t *det,
    const size_t nocca,
    const size_t noccb
    )
{
  for (int i = 0; i < DET_LEN; i++) {
    det[i] = 0;
  }
  u_int64_t mask = 1;
  for (int i = 0; i < nocca; i++) {
    int spin_occ = 2*occa[i];
    int det_ind = spin_occ / DET_SIZE;
    int det_pos = spin_occ % DET_SIZE;
    /*printf("alpha : %d %d %d\n", spin_occ, det_ind, det_pos);*/
    det[det_ind] |= (mask << det_pos);
  }
  for (int i = 0; i < noccb; i++) {
    int spin_occ = 2*occb[i] + 1;
    int det_ind = spin_occ / DET_SIZE;
    int det_pos = spin_occ % DET_SIZE;
    /*printf("beta:  %d %d %d\n", spin_occ, det_ind, det_pos);*/
    det[det_ind] |= (mask << det_pos);
  }
  /*printf("end\n");*/
}

int count_set_bits(const u_int64_t *det)
{
  int nset = 0;
  for (int i = 0; i < DET_LEN; i++) {
    nset += __builtin_popcountll(det[i]);
  }
  return nset;
}
int count_set_bits_single(const u_int64_t det)
{
  return __builtin_popcountll(det);
}

int get_excitation_level(
    const u_int64_t *deta,
    const u_int64_t *detb)
{
  int excit_level = 0;
  for (int i = 0; i < DET_LEN; i++) {
      excit_level += count_set_bits_single(deta[i]^detb[i]);
      /*printf("%d %d %llu %llu %d\n", i, excit_level, deta[i], detb[i], count_set_bits_single(deta[i]^detb[i]));*/
  }
  return excit_level / 2;
}

void decode_det(
    const u_int64_t *det,
    int *occs,
    const size_t nel
    )
{
  // dumb for the moment.
  int nset = 0;
  u_int64_t mask = 1;
  for (int det_ind = 0; det_ind < DET_LEN; det_ind++) {
    for (int bit_pos=0; bit_pos < DET_SIZE; bit_pos++) {
      if (det[det_ind] & (mask<<bit_pos)) {
        occs[nset] = bit_pos + det_ind * DET_SIZE;
        nset++;
      }
      if (nset == nel)
        break;
    }
    if (nset == nel)
      break;
  }
}

/*bool is_set(*/
    /*u_int64_t *det,*/
    /*int loc*/
    /*)*/
/*{*/
  /*bool set = false;*/
  /*for (int i = 0; i < DET_LEN; i++) {*/
    /*set &= det & (loc << diff[0])*/
  /*}*/
/*}*/

void build_set_mask(
    u_int64_t *mask,
    int det_loc, // bitstring array index
    int det_ind  // bit index
    )
{
  u_int64_t all_set = 0xFFFFFFFFFFFFFFFF;
  u_int64_t one = 1;
  for (int i = 0; i < det_loc; i++) {
    mask[i] = all_set;
  }
  mask[det_loc] = (one << det_ind) - one;
}

// should have used C++
void bitwise_and(
    u_int64_t *deta,
    u_int64_t *detb,
    u_int64_t *result
    )
{
  for (int i = 0; i < DET_LEN; i++) {
    result[i] = deta[i] & detb[i];
  }
}
void bitwise_subtract(
    u_int64_t *deta,
    u_int64_t *detb,
    u_int64_t *result
    )
{
  for (int i = 0; i < DET_LEN; i++) {
    result[i] = deta[i] - detb[i];
  }
}

// < bra | a^ i | ket >
void get_ia(
    u_int64_t *det_bra,
    u_int64_t *det_ket,
    int* ia)
{
  int diff[2];
  u_int64_t delta[2];
  for (int i = 0; i < DET_LEN; i++) {
    delta[i] = det_bra[i] ^ det_ket[i];
  }
  decode_det(delta, diff, 2);
  u_int64_t loc = 1;
  int a = 0;
  int bit_pos = diff[0] / DET_SIZE;
  if (det_ket[bit_pos] & (loc << diff[0])) {
    ia[0] = diff[0];
    ia[1] = diff[1];
  } else {
    ia[0] = diff[1];
    ia[1] = diff[0];
  }
}

// phase( a^i|ket> )
int get_perm_ia(
    u_int64_t *det_ket,
    int i,
    int a)
{
  u_int64_t and_mask[DET_LEN], mask_i[DET_LEN], mask_a[DET_LEN];
  u_int64_t occ_to_count[DET_LEN];
  u_int64_t loc = 1;
  for (int i = 0; i < DET_LEN; i++) {
    and_mask[i] = 0; // all bits set to 0
    mask_i[i] = 0; // all bits set to 0
    mask_a[i] = 0; // all bits set to 0
    occ_to_count[i] = 0;
  }
  // check bit a is occupied or bit i is unoccupied.
  // else just count set bits between i and a.
  int det_ind_a = a / DET_SIZE;
  int det_ind_i = i / DET_SIZE;
  int det_pos_a = a % DET_SIZE;
  int det_pos_i = i % DET_SIZE;
  /*printf("index: %d %d %d %d %d %d\n", i, a, det_ind_a, det_ind_i, det_pos_a, det_pos_i);*/
  if (a == i) {
    return 1;
  } else {
    if ((det_ket[det_ind_a] & (loc << det_pos_a)) || !(det_ket[det_ind_i] & (loc << det_pos_i))) {
      return 0;
    } else {
      if (i > a) {
        build_set_mask(mask_a, det_ind_a, det_pos_a+1);
        build_set_mask(mask_i, det_ind_i, det_pos_i);
        bitwise_subtract(mask_i, mask_a, and_mask);
      } else {
        build_set_mask(mask_a, det_ind_a, det_pos_a);
        build_set_mask(mask_i, det_ind_i, det_pos_i+1);
        bitwise_subtract(mask_a, mask_i, and_mask);
      }
    }
    bitwise_and(det_ket, and_mask, occ_to_count);
    /*printf("%d %d %d\n", and_mask[0], det_ket[0], occ_to_count[0]);*/
    /*printf("masks : %llu %llu %llu %llu %d %d\n", mask_i[0], mask_a[0], and_mask[0], occ_to_count[0], det_pos_i, det_pos_a);*/
    if (count_set_bits(occ_to_count)%2 == 0) {
      return 1;
    } else {
      return -1;
    }
  }
}
