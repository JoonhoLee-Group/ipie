#ifndef _IPIE_CUSTOM_TYPES_H
#define _IPIE_CUSTOM_TYPES_H
#include <complex>

#include "config.h"
namespace ipie {

struct energy_t {
    energy_t() : etot{0}, e1b{0}, e2b{0} {};
    energy_t(ipie::complex_t e, ipie::complex_t t, ipie::complex_t v) : etot{e}, e1b{t}, e2b{v} {};
    energy_t &operator+=(const energy_t &other);
    energy_t &operator*=(const double &scale);
    energy_t &operator*=(const ipie::complex_t &scale);
    // approximate equality
    bool operator==(const energy_t &other);
    friend std::ostream &operator<<(std::ostream &os, energy_t &energy);
    std::complex<double> etot;
    std::complex<double> e1b;
    std::complex<double> e2b;
};

}  // namespace ipie
#endif