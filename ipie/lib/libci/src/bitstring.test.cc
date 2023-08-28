#include "bitstring.h"

#include "gtest/gtest.h"

TEST(bitstring, constructor) {
    auto bs = ipie::BitString(100);
    ASSERT_EQ(bs.num_words, 2);
    ASSERT_EQ(bs.bitstring.size(), 2);
    ASSERT_EQ(bs.bitstring[0], 0);
    ASSERT_EQ(bs.bitstring[1], 0);
    ASSERT_EQ(bs.num_bits, 100);
}

TEST(bitstring, copy) {
    auto bs = ipie::BitString(100);
    bs.bitstring[0] = ~0ULL;
    auto other = bs;
    ASSERT_EQ(other.bitstring[0], ~0ULL);
}

TEST(bitstring, copy_assign) {
    auto bs = ipie::BitString(100);
    bs.bitstring[0] = ~0ULL;
    auto other(bs);
    ASSERT_EQ(other.bitstring[0], ~0ULL);
}

TEST(bitstring, xor_equal) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    a.bitstring[0] = ~0ULL;
    a.bitstring[1] = 0ULL;
    b ^= a;
    ASSERT_EQ(b.bitstring[0], ~0ULL);
    ASSERT_EQ(b.bitstring[1], 0ULL);
}

TEST(bitstring, and_equal) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    a.bitstring[0] = ~0ULL;
    b.bitstring[1] = ~0ULL;
    b &= a;
    ASSERT_EQ(b.bitstring[0], 0ULL);
    ASSERT_EQ(b.bitstring[1], 0ULL);
    a.bitstring[1] = ~0ULL;
    b.bitstring[1] = ~0ULL;
    ASSERT_EQ(a.bitstring[1], ~0ULL);
    ASSERT_EQ(b.bitstring[1], ~0ULL);
    b &= a;
    ASSERT_EQ(b.bitstring[1], ~0ULL);
}

TEST(bitstring, or_equal) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    a.bitstring[0] = ~0ULL;
    b.bitstring[1] = ~0ULL;
    b |= a;
    ASSERT_EQ(b.bitstring[0], ~0ULL);
    ASSERT_EQ(b.bitstring[1], ~0ULL);
}

TEST(bitstring, subscript) {
    auto a = ipie::BitString(100);
    ASSERT_EQ(a[0], 0ULL);
    ASSERT_EQ(a[0], 0ULL);
    a[0] = 1ULL << 32;
    ASSERT_EQ(a[0], 1ULL << 32);
    ASSERT_EQ(a[1], 0ULL);
}

TEST(bitstring, not_equal) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    a[0] = 1ULL << 32;
    ASSERT_NE(a, b);
}

TEST(bitstring, equal) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    a[0] = 1ULL << 32;
    b[0] = 1ULL << 32;
    ASSERT_EQ(a, b);
}

TEST(bitstring, build_set_mask) {
    auto mask = ipie::BitString(100);
    build_set_mask(67, mask);
    for (int i = 0; i < 67; i++) {
        ASSERT_EQ(mask.is_set(i), true);
    }
    for (int i = 67; i < 128; i++) {
        ASSERT_EQ(mask.is_set(i), false);
    }
}

TEST(bitstring, encode_bits) {
    auto a = ipie::BitString(100);
    std::vector<int> bit_indxs = {1, 37, 66, 73};
    a.encode_bits(bit_indxs);
    for (auto i : bit_indxs) {
        ASSERT_EQ(a.is_set(i), true);
    }
}
TEST(bitstring, decode_bits) {
    auto a = ipie::BitString(100);
    std::vector<int> bit_indxs = {1, 37, 66, 73};
    a.encode_bits(bit_indxs);
    std::vector<int> test(4);
    a.decode_bits(test);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(test[i], bit_indxs[i]);
    }
}

TEST(bitstring, count_set_bits) {
    auto a = ipie::BitString(100);
    std::vector<int> bit_indxs = {1, 37, 66, 73};
    a.encode_bits(bit_indxs);
    ASSERT_EQ(a.count_set_bits(), 4);
    a[0] = 0xFFFFFFFFFFFFFFFF;
    a[1] = 0xFFFFFFFFFFFFFFFF;
    ASSERT_EQ(a.count_set_bits(), 128);
}

TEST(bitstring, count_difference) {
    auto a = ipie::BitString(100);
    auto b = ipie::BitString(100);
    std::vector<int> bit_indxs = {1, 37, 66, 73};
    a.encode_bits(bit_indxs);
    b.encode_bits(bit_indxs);
    ASSERT_EQ(a.count_difference(b), 0);
    std::vector<int> other_indxs = {2, 37, 66, 73};
    b.encode_bits(other_indxs);
    ASSERT_EQ(a.count_difference(b), 1);
    other_indxs = {2, 38, 66, 73};
    b.encode_bits(other_indxs);
    ASSERT_EQ(a.count_difference(b), 2);
    other_indxs = {2, 38, 69, 73};
    b.encode_bits(other_indxs);
    ASSERT_EQ(a.count_difference(b), 3);
    other_indxs = {2, 38, 69, 74};
    b.encode_bits(other_indxs);
    ASSERT_EQ(a.count_difference(b), 4);
}

TEST(bitstring, clear_bits) {
    auto a = ipie::BitString(100);
    a[0] = 0xFFFFFFFFFFFFFFFF;
    a[1] = 0xFFFFFFFFFFFFFFFF;
    a.clear_bits();
    ASSERT_EQ(a.count_set_bits(), 0);
}

TEST(bitstring, set_clear_bit) {
    auto a = ipie::BitString(100);
    for (size_t i = 0; i < 100; i++) {
        a.set_bit(i);
        ASSERT_EQ(a.count_set_bits(), i + 1);
    }
    for (size_t i = 0; i < 100; i++) {
        a.clear_bit(i);
        ASSERT_EQ(a.count_set_bits(), 100 - (i + 1));
    }
}