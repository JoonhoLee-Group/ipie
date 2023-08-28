#include "excitations.h"

#include <vector>

namespace ipie {

Excitation::Excitation(size_t max_ex) : max_excit(max_ex), from(max_ex), to(max_ex) {
}

std::ostream &operator<<(std::ostream &os, const Excitation &excit) {
    for (size_t e = 0; e < excit.max_excit; e++) {
        os << excit.from[e] << " ";
    }
    os << " -> ";
    for (size_t e = 0; e < excit.max_excit; e++) {
        os << excit.to[e] << " ";
    }
    os << " \n";
    return os;
}

void decode_single_excitation(BitString &bra, BitString &ket, Excitation &ia) {
    std::vector<int> diff_bits(2);
    BitString delta(bra);
    delta ^= ket;
    delta.decode_bits(diff_bits);
    if (ket.is_set(diff_bits[0])) {
        ia.from[0] = diff_bits[0];
        ia.to[0] = diff_bits[1];
    } else {
        ia.from[0] = diff_bits[1];
        ia.to[0] = diff_bits[0];
    }
}
void decode_double_excitation(BitString &bra, BitString &ket, Excitation &ijab) {
    // delta = bra ^ ket.
    BitString delta(bra);
    delta ^= ket;
    std::vector<int> diff_bits(4);
    delta.decode_bits(diff_bits);
    size_t ifrom = 0, ito = 0;
    for (size_t i = 0; i < 4; i++) {
        // |bra> = a^ b^ i j | ket>
        // from should store ij
        // to should store ab
        if (ket.is_set(diff_bits[i])) {
            ijab.from[ifrom] = diff_bits[i];
            ifrom++;
        } else {
            ijab.to[ito] = diff_bits[i];
            ito++;
        }
    }
}

int single_excitation_permutation(BitString &ket, Excitation &ia) {
    BitString and_mask(ket.num_bits), mask_i(ket.num_bits), mask_a(ket.num_bits);
    BitString occ_to_count(ket);
    // check bit a is occupied or bit i is unoccupied.
    // else just count set bits between i and a.
    int i = ia.from[0];
    int a = ia.to[0];
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

int double_excitation_permutation(BitString &ket, Excitation &ijab) {
    BitString mask(ket.num_bits);
    BitString tmp(ket);
    BitString occ_to_count(ket);
    // (-1)^{perm} a^b^ji|ket>
    // i
    build_set_mask(ijab.from[0], mask);
    tmp &= mask;
    size_t num_set = tmp.count_set_bits();
    tmp.clear_bit(ijab.from[0]);
    mask.clear_bits();
    build_set_mask(ijab.from[0], mask);
    // j
    build_set_mask(ijab.from[1], mask);
    tmp &= mask;
    num_set += tmp.count_set_bits();
    tmp.clear_bit(ijab.from[1]);
    mask.clear_bits();
    // b
    build_set_mask(ijab.to[1], mask);
    tmp &= mask;
    num_set += tmp.count_set_bits();
    tmp.clear_bit(ijab.from[1]);
    mask.clear_bits();
    // a
    build_set_mask(ijab.to[0], mask);
    tmp &= mask;
    num_set += tmp.count_set_bits();
    tmp.clear_bit(ijab.from[0]);
    mask.clear_bits();
    return num_set % 2 ? 1 : -1;
}
}  // namespace ipie