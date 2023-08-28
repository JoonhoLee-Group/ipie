
#include "excitations.h"

#include "gtest/gtest.h"

TEST(excitations, decode_single_excitation) {
    auto bra = ipie::BitString(100);
    auto ket = ipie::BitString(100);
    bra.set_bit(0);
    bra.set_bit(73);
    ket.set_bit(0);
    ket.set_bit(99);
    ipie::Excitation ia(1);
    decode_single_excitation(bra, ket, ia);
    ASSERT_EQ(ia.from[0], 99);
    ASSERT_EQ(ia.to[0], 73);
}

TEST(excitations, single_excitation_permutation) {
    auto bra = ipie::BitString(100);
    auto ket = ipie::BitString(100);
    bra.set_bit(0);
    bra.set_bit(1);
    ket.set_bit(0);
    ket.set_bit(2);
    ipie::Excitation ia(1);
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

TEST(excitations, decode_double_excitation) {
    auto bra = ipie::BitString(100);
    auto ket = ipie::BitString(100);
    bra.set_bit(0);
    bra.set_bit(33);
    bra.set_bit(73);
    bra.set_bit(94);
    ket.set_bit(0);
    ket.set_bit(33);
    ket.set_bit(74);
    ket.set_bit(99);
    ipie::Excitation ijab(2);
    decode_double_excitation(bra, ket, ijab);
    ASSERT_EQ(ijab.from[0], 74);
    ASSERT_EQ(ijab.from[1], 99);
    ASSERT_EQ(ijab.to[0], 73);
    ASSERT_EQ(ijab.to[1], 94);
}