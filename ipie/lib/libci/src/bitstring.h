#ifndef _BITSTRING_H
#define _BITSTRING_H

#include <iostream>
#include <sstream>
#include <vector>

namespace ipie {

// Lightweight structure for handling arrays of 64-bit ints which we use to represent determinants.
struct BitString {
    size_t num_bits;
    size_t num_words;

    std::vector<uint64_t> bitstring;

    BitString(size_t num_bits);

    BitString &operator^=(const BitString &other);
    BitString &operator|=(const BitString &other);
    BitString &operator&=(const BitString &other);
    BitString operator-(const BitString &other);

    bool operator==(const BitString &other) const;
    bool operator!=(const BitString &other) const;
    uint64_t &operator[](const size_t indx);

    void encode_bits(std::vector<int> &set_bits);
    void decode_bits(std::vector<int> &set_bits);
    void clear_bits();
    void clear_bit(const size_t bit_indx);
    bool is_set(const size_t bit_indx) const;
    void set_bit(const size_t bit_indx);
    void set_bits(const std::vector<size_t> &bit_indx);
    size_t count_set_bits();
    size_t count_difference(const BitString &other);
};

std::ostream &operator<<(std::ostream &os, const BitString &bs);
void build_set_mask(size_t bit_indx, BitString &mask);

}  // namespace ipie

#endif