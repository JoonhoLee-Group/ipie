#ifndef _EXCITATIONS_H
#define _EXCITATIONS_H

#include <iostream>

#include "bitstring.h"

namespace ipie {

struct Excitation {
    size_t max_excit;
    Excitation(size_t max_ex);
    std::vector<size_t> from;
    std::vector<size_t> to;
};
void decode_single_excitation(BitString& bra, BitString& ket, Excitation& ia);
void decode_double_excitation(BitString& bra, BitString& ket, Excitation& ijab);
int single_excitation_permutation(BitString& ket, Excitation& ia);

}  // namespace ipie
#endif