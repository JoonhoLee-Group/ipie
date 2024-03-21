#ifndef _DETERMINANT_H
#define _DETERMINANT_H

#include <iostream>
#include <sstream>
#include <vector>

#include "bitstring.h"

namespace ipie {
struct Determinant {
    Determinant(BitString a, BitString b) : alpha(std::move(a)), beta(std::move(b)) {
        num_spatial = a.num_bits;
        num_alpha = a.count_set_bits();
        num_beta = b.count_set_bits();
    }
    bool operator==(const Determinant &other) const;
    size_t count_difference(const Determinant &other) const;
    inline size_t count_set_bits(size_t spin) {
        if (spin == 0) {
            return alpha.count_set_bits();
        } else {
            return beta.count_set_bits();
        }
    }
    inline bool is_set(const size_t bit_indx, size_t spin) {
        if (spin == 0) {
            return alpha.is_set(bit_indx);
        } else {
            return beta.is_set(bit_indx);
        }
    }
    inline void set_bit(const size_t bit_indx, size_t spin) {
        if (spin == 0) {
            alpha.set_bit(bit_indx);
        } else {
            beta.set_bit(bit_indx);
        }
    }
    inline void clear_bit(const size_t bit_indx, size_t spin) {
        if (spin == 0) {
            alpha.clear_bit(bit_indx);
        } else {
            beta.clear_bit(bit_indx);
        }
    }
    inline BitString &operator[](const size_t spin) {
        if (spin == 0) {
            return alpha;
        } else {
            return beta;
        }
    }
    BitString alpha;
    BitString beta;
    size_t num_spatial;
    size_t num_alpha;
    size_t num_beta;
};
std::ostream &operator<<(std::ostream &os, const Determinant &det);
struct DeterminantHasher {
    uint64_t operator()(const Determinant &det) const {
        // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
        // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key/12996028#12996028
        uint64_t seed = det.alpha.bitstring.size() + det.beta.bitstring.size();
        // just make this an array that is twice as long
        for (auto b : det.alpha.bitstring) {
            b = (b ^ (b >> 30)) * uint64_t{0XBF58476D1CE4E5B9};
            b = (b ^ (b >> 27)) * uint64_t{0X94D049BB133111EB};
            b = b ^ (b >> 31);
            // This magic number is related to the integral part of the golden ratio
            // https://softwareengineering.stackexchange.com/questions/402542/where-do-magic-hashing-constants-like-0x9e3779b9-and-0x9e3779b1-come-from
            seed ^= b + uint64_t{0X9E3779B97F4A7C15} + (seed << 6) + (seed >> 2);
        }
        for (auto b : det.beta.bitstring) {
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

}  // namespace ipie

#endif