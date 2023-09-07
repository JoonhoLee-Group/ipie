#ifndef _EXCITATIONS_H
#define _EXCITATIONS_H

#include <iostream>

#include "bitstring.h"

namespace ipie {

struct Excitation {
    size_t max_excit;
    Excitation(size_t max_ex);
    Excitation(std::vector<size_t>, std::vector<size_t>);
    std::vector<size_t> from;
    std::vector<size_t> to;
};

std::ostream& operator<<(std::ostream& os, const Excitation& excit);

void decode_single_excitation(const BitString& bra, const BitString& ket, Excitation& ia);
void decode_double_excitation(const BitString& bra, const BitString& ket, Excitation& ijab);
int single_excitation_permutation(const BitString& ket, Excitation& ia);
int double_excitation_permutation(const BitString& ket, Excitation& ijab);

}  // namespace ipie
#endif