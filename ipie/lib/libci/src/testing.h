#ifndef _TESTING_H
#define _TESTING_H

#include <iostream>
#include <sstream>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
ipie::Hamiltonian build_test_hamiltonian(size_t num_orb);
ipie::Wavefunction build_test_wavefunction(size_t num_dets, size_t num_bits);
}  // namespace ipie

#endif