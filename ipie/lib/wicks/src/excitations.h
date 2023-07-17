#ifndef _EXCITATIONS_H
#define _EXCITATIONS_H

#include <iostream>

#include "bitstring.h"

namespace ipie {

void decode_single_excitation(BitString& bra, BitString& ket, std::vector<int>& ia);
int single_excitation_permutation(BitString& ket, std::vector<int>& ia);

}  // namespace ipie
#endif