#include "testing.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace ipie {
ipie::Hamiltonian build_test_hamiltonian(size_t num_orb) {
    std::vector<ipie::complex_t> h1e(num_orb * num_orb);
    std::vector<ipie::complex_t> h2e(num_orb * num_orb * num_orb * num_orb);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < h1e.size(); i++) {
        h1e[i] = ipie::complex_t(distribution(generator), distribution(generator));
    }
    for (size_t i = 0; i < h2e.size(); i++) {
        h2e[i] = ipie::complex_t(distribution(generator), distribution(generator));
    }
    ipie::complex_t e0 = ipie::complex_t(distribution(generator), 0.0);
    return ipie::Hamiltonian(h1e, h2e, e0, num_orb);
}

ipie::Wavefunction build_test_wavefunction(size_t num_dets_max, size_t num_spat_max) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> det_dist(1, num_dets_max);
    size_t ndets = (size_t)det_dist(generator);
    std::uniform_int_distribution<int> spat_dist(1, num_spat_max);
    std::uniform_real_distribution<double> c_dist(0, 1);
    int num_spin = 2 * spat_dist(generator);
    ipie::det_map dmap;
    size_t num_elec;
    std::uniform_int_distribution<int> elec_dist(1, (int)num_spin);
    num_elec = (size_t)elec_dist(generator);
    std::vector<size_t> range(num_spin);
    for (size_t idet = 0; idet < ndets; idet++) {
        auto a = ipie::BitString(num_spin);
        std::iota(range.begin(), range.end(), 0);
        std::shuffle(range.begin(), range.end(), generator);
        for (size_t i = 0; i < num_elec; i++) {
            a.set_bit(range[i]);
        }
        std::complex<double> coeff = std::complex<double>(c_dist(generator), 0.0);
        if (dmap.size() == 0 || dmap.find(a) == dmap.end()) {
            // we don't want duplicate keys.
            dmap.insert({a, coeff});
        }
    }
    ipie::Wavefunction wfn(dmap);
    return wfn;
}

ipie::Wavefunction build_test_wavefunction_restricted(size_t num_dets_max, size_t num_spat_max) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> det_dist(1, num_dets_max);
    size_t ndets = (size_t)det_dist(generator);
    std::uniform_int_distribution<int> spat_dist(1, num_spat_max);
    std::uniform_real_distribution<double> c_dist(0, 1);
    int num_spin = 2 * spat_dist(generator);
    ipie::det_map dmap;
    std::vector<ipie::BitString> dets;
    size_t num_alpha;
    std::uniform_int_distribution<int> elec_dist(1, (int)num_spin / 2);
    num_alpha = (size_t)elec_dist(generator);
    std::vector<size_t> range(num_spin / 2);
    for (size_t idet = 0; idet < ndets; idet++) {
        auto a = ipie::BitString(num_spin);
        std::iota(range.begin(), range.end(), 0);
        std::shuffle(range.begin(), range.end(), generator);
        for (size_t i = 0; i < num_alpha; i++) {
            a.set_bit(2 * range[i]);
        }
        std::iota(range.begin(), range.end(), 0);
        std::shuffle(range.begin(), range.end(), generator);
        for (size_t i = 0; i < num_alpha; i++) {
            a.set_bit(2 * range[i] + 1);
        }
        std::complex<double> coeff = std::complex<double>(c_dist(generator), 0.0);
        if (dmap.size() == 0 || dmap.find(a) == dmap.end()) {
            // we don't want duplicate keys.
            dmap.insert({a, coeff});
        }
    }
    ipie::Wavefunction wfn(dmap);
    return wfn;
}
}  // namespace ipie