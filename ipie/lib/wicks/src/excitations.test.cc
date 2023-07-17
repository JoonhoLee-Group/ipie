#include "excitations.h"

#include "gtest/gtest.h"

#include "bitstring.h"
#include "excitations.h"

TEST(excitations, decode_single_excitation) {
    auto bra = ipie::BitString(100);
    auto ket = ipie::BitString(100);
    bra.set_bit(0);
    bra.set_bit(73);
    ket.set_bit(0);
    ket.set_bit(99);
    std::vector<int> ia(2);
    decode_single_excitation(bra, ket, ia);
    ASSERT_EQ(ia[0], 99);
    ASSERT_EQ(ia[1], 73);
}

TEST(excitations, single_excitation_permutation) {
    auto bra = ipie::BitString(100);
    auto ket = ipie::BitString(100);
    bra.set_bit(0);
    bra.set_bit(1);
    ket.set_bit(0);
    ket.set_bit(2);
    std::vector<int> ia(2);
    decode_single_excitation(bra, ket, ia);
    int perm = single_excitation_permutation(ket, ia);
    ASSERT_EQ(perm, 1);
    bra.clear_bits();
    ket.clear_bits();
    bra.set_bit(1);
    bra.set_bit(3);
    ket.set_bit(0);
    ket.set_bit(1);
    decode_single_excitation(bra, ket, ia);
    perm = single_excitation_permutation(ket, ia);
    ASSERT_EQ(perm, -1);
    // exit case
    bra.clear_bits();
    ket.clear_bits();
    bra.set_bit(1);
    bra.set_bit(3);
    ket.set_bit(1);
    ket.set_bit(3);
    decode_single_excitation(bra, ket, ia);
    perm = single_excitation_permutation(ket, ia);
    ASSERT_EQ(perm, 1);
    // more complicated case i > a
    bra.clear_bits();
    ket.clear_bits();
    bra.set_bit(0);
    bra.set_bit(1);
    bra.set_bit(11);
    bra.set_bit(13);
    ket.set_bit(0);
    ket.set_bit(1);
    ket.set_bit(13);
    ket.set_bit(14);
    decode_single_excitation(bra, ket, ia);
    perm = single_excitation_permutation(ket, ia);
    ASSERT_EQ(perm, -1);
    // more complicated case a > i
    bra.clear_bits();
    ket.clear_bits();
    bra.set_bit(0);
    bra.set_bit(1);
    bra.set_bit(11);
    bra.set_bit(13);
    ket.set_bit(0);
    ket.set_bit(1);
    ket.set_bit(11);
    ket.set_bit(15);
    decode_single_excitation(bra, ket, ia);
    perm = single_excitation_permutation(ket, ia);
    ASSERT_EQ(perm, 1);
}