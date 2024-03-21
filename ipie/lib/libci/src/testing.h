#ifndef _TESTING_H
#define _TESTING_H

#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
// static bool test_rng_initialized;
// static std::mt19937_64 test_rng;
ipie::Hamiltonian build_test_hamiltonian(size_t num_orb);
ipie::Wavefunction build_test_wavefunction(size_t num_dets, size_t num_bits, size_t num_alpha, size_t num_beta);
ipie::Wavefunction build_ci_singles(size_t num_spat_max, size_t num_alpha, size_t num_beta);
}  // namespace ipie

#endif