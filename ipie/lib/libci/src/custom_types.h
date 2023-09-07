#ifndef _IPIE_CUSTOM_TYPES_H
#define _IPIE_CUSTOM_TYPES_H
#include <complex>

#include "config.h"
namespace ipie {

struct new_energy_t {
    new_energy_t(ipie::complex_t e, ipie::complex_t t, ipie::complex_t v) : etot{e}, e1b{t}, e2b{v} {};
    new_energy_t &operator+=(const new_energy_t &other);
    new_energy_t &operator*=(const double &scale);
    new_energy_t &operator*=(const ipie::complex_t &scale);
    friend std::ostream &operator<<(std::ostream &os, new_energy_t &energy);
    std::complex<double> etot;
    std::complex<double> e1b;
    std::complex<double> e2b;
};

}  // namespace ipie
#endif