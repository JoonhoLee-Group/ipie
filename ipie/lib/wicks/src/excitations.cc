#include <vector>

#include "bitstring.h"

namespace ipie {
void decode_single_excitation(BitString &bra, BitString &ket, std::vector<int> &ia) {
    std::vector<int> diff_bits(2);
    BitString delta(bra);
    delta ^= ket;
    delta.decode_bits(diff_bits);
    if (ket.is_set(diff_bits[0])) {
        ia[0] = diff_bits[0];
        ia[1] = diff_bits[1];
    } else {
        ia[0] = diff_bits[1];
        ia[1] = diff_bits[0];
    }
}
int single_excitation_permutation(BitString &ket, std::vector<int> &ia) {
    BitString and_mask(ket.num_bits), mask_i(ket.num_bits), mask_a(ket.num_bits);
    BitString occ_to_count(ket);
    // check bit a is occupied or bit i is unoccupied.
    // else just count set bits between i and a.
    int i = ia[0];
    int a = ia[1];
    if (a == i) {
        return 1;
    } else {
        if (ket.is_set(a) || (!ket.is_set(i))) {
            return 0;
        } else {
            if (i > a) {
                build_set_mask(a + 1, mask_a);
                build_set_mask(i, mask_i);
                and_mask = mask_i - mask_a;
            } else {
                build_set_mask(a, mask_a);
                build_set_mask(i + 1, mask_i);
                and_mask = mask_a - mask_i;
            }
        }
        occ_to_count &= and_mask;
        if (occ_to_count.count_set_bits() % 2 == 0) {
            return 1;
        } else {
            return -1;
        }
    }
}
}  // namespace ipie