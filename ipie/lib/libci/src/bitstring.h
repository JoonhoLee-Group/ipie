#ifndef _BITSTRING_H
#define _BITSTRING_H

#include <iostream>
#include <sstream>
#include <vector>

namespace ipie {

// Lightweight structure for handling arrays of 64-bit ints which we use to represent determinants.
// Heavily inspired by Stim simd_bits
struct BitString {
    size_t num_bits;
    size_t num_words;

    std::vector<uint64_t> bitstring;

    BitString() = default;
    BitString(size_t num_bits);

    BitString &operator^=(const BitString &other);
    BitString &operator|=(const BitString &other);
    BitString &operator&=(const BitString &other);
    BitString operator-(const BitString &other);

    bool operator==(const BitString &other) const;
    bool operator!=(const BitString &other) const;
    uint64_t &operator[](const size_t indx);

    void encode_bits(std::vector<size_t> &set_bits);
    void decode_bits(std::vector<size_t> &set_bits) const {
        size_t num_set = 0;
        uint64_t mask = 1;
        for (size_t w = 0; w < bitstring.size(); w++) {
            for (int bit_pos = 0; bit_pos < 64; bit_pos++) {
                if (bitstring[w] & (mask << bit_pos)) {
                    set_bits[num_set] = bit_pos + w * 64;
                    num_set++;
                }
                if (num_set == set_bits.size())
                    break;
            }
            if (num_set == set_bits.size())
                break;
        }
    }
    void clear_bits();
    void clear_bit(const size_t bit_indx);
    inline bool is_set(const size_t bit_indx) const {
        size_t word = bit_indx / 64;
        int offset = bit_indx - word * 64;
        uint64_t one = 1;
        return bitstring[word] & (one << offset);
    };
    // void set_bit(const size_t bit_indx);
    inline void set_bit(const size_t bit_indx) {
        size_t word = bit_indx / 64;
        int offset = bit_indx - word * 64;
        uint64_t one{1};
        bitstring[word] |= (one << offset);
    }
    void set_bits(const std::vector<size_t> &bit_indx);
    size_t count_set_bits() const;
    size_t count_difference(const BitString &other) const;
};

struct BitStringHasher {
    uint64_t operator()(const BitString &bs) const {
        // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
        // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key/12996028#12996028
        uint64_t seed = bs.bitstring.size();
        for (auto b : bs.bitstring) {
            b = (b ^ (b >> 30)) * uint64_t{0XBF58476D1CE4E5B9};
            b = (b ^ (b >> 27)) * uint64_t{0X94D049BB133111EB};
            b = b ^ (b >> 31);
            // This magic number is related to the integral part of the golden ratio
            // https://softwareengineering.stackexchange.com/questions/402542/where-do-magic-hashing-constants-like-0x9e3779b9-and-0x9e3779b1-come-from
            seed ^= b + uint64_t{0X9E3779B97F4A7C15} + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::ostream &operator<<(std::ostream &os, const BitString &bs);
void build_set_mask(size_t bit_indx, BitString &mask);

}  // namespace ipie

#endif