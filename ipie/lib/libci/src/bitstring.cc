#include "bitstring.h"

#include <sstream>

namespace ipie {

BitString::BitString() : num_bits(0) {
}

BitString::BitString(size_t nbits) : num_bits(nbits) {
    num_words = (nbits + 64 - 1) / 64;
    bitstring.resize(num_words);
}

uint64_t& BitString::operator[](const size_t indx) {
    return bitstring[indx];
}

BitString& BitString::operator^=(const BitString& other) {
    for (size_t w = 0; w < num_words; w++) {
        bitstring[w] ^= other.bitstring[w];
    }
    return *this;
}

// This is wrong, only for subtracting 1?
BitString BitString::operator-(const BitString& other) {
    BitString result(other.num_bits);
    for (size_t w = 0; w < num_words; w++) {
        result[w] = bitstring[w] - other.bitstring[w];
    }
    return result;
}

BitString& BitString::operator&=(const BitString& other) {
    for (size_t w = 0; w < num_words; w++) {
        bitstring[w] &= other.bitstring[w];
    }
    return *this;
}

BitString& BitString::operator|=(const BitString& other) {
    for (size_t w = 0; w < num_words; w++) {
        bitstring[w] |= other.bitstring[w];
    }
    return *this;
}
bool BitString::operator==(const BitString& other) const {
    return bitstring == other.bitstring;
}

bool BitString::operator!=(const BitString& other) const {
    return bitstring != other.bitstring;
}

// Build a mask with all bits before bit indx set
void BitString::encode_bits(std::vector<size_t>& set_bits) {
    clear_bits();
    uint64_t mask = 1;
    for (auto sb : set_bits) {
        size_t word_indx = sb / 64;
        size_t bit_pos = sb % 64;
        bitstring[word_indx] |= (mask << bit_pos);
    }
}

void BitString::clear_bits() {
    for (size_t i = 0; i < num_words; i++) {
        bitstring[i] = 0ULL;
    }
}

void BitString::decode_bits(std::vector<size_t>& set_bits) const {
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

size_t BitString::count_set_bits() const {
    size_t num_set = 0;
    for (size_t w = 0; w < bitstring.size(); w++) {
        num_set += __builtin_popcountll(bitstring[w]);
    }
    return num_set;
}

size_t BitString::count_difference(const BitString& other) const {
    size_t num_set = 0;
    for (size_t w = 0; w < bitstring.size(); w++) {
        num_set += __builtin_popcountll(bitstring[w] ^ other.bitstring[w]);
    }
    return num_set / 2;
}

bool BitString::is_set(const size_t indx) const {
    size_t word = indx / 64;
    int offset = indx - word * 64;
    uint64_t one = 1;
    return bitstring[word] & (one << offset);
}

void BitString::set_bit(const size_t indx) {
    size_t word = indx / 64;
    int offset = indx - word * 64;
    uint64_t one = 1;
    bitstring[word] |= (one << offset);
}

void BitString::set_bits(const std::vector<size_t>& bit_indxs) {
    for (auto i : bit_indxs) {
        set_bit(i);
    }
}

void BitString::clear_bit(const size_t indx) {
    size_t word = indx / 64;
    int offset = indx - word * 64;
    uint64_t one = 1;
    bitstring[word] ^= (one << offset);
}

// all ones before bit_indx
void build_set_mask(size_t bit_indx, BitString& mask) {
    uint64_t all_set = 0xFFFFFFFFFFFFFFFF;
    uint64_t one = 1;
    size_t word_indx = bit_indx / 64;
    size_t word_bit_indx = bit_indx - word_indx * 64;
    for (size_t w = 0; w < word_indx; w++) {
        mask[w] = all_set;
    }
    mask[word_indx] = (one << word_bit_indx) - one;
}

std::ostream& operator<<(std::ostream& os, const BitString& bs) {
    for (size_t bit = 0; bit < bs.num_bits; bit++) {
        os << bs.is_set(bit);
    }
    return os;
}

}  // namespace ipie