#include "custom_types.h"

#include <iomanip>
#include <iostream>
#include <ostream>

namespace ipie {
energy_t &energy_t::operator+=(const energy_t &other) {
    etot += other.etot;
    e1b += other.e1b;
    e2b += other.e2b;
    return *this;
}
energy_t &energy_t::operator*=(const double &scale) {
    etot *= ipie::complex_t{scale};
    e1b *= ipie::complex_t{scale};
    e2b *= ipie::complex_t{scale};
    return *this;
}
energy_t &energy_t::operator*=(const ipie::complex_t &scale) {
    etot *= scale;
    e1b *= scale;
    e2b *= scale;
    return *this;
}

energy_t &energy_t::operator/=(const ipie::complex_t &scale) {
    etot /= scale;
    e1b /= scale;
    e2b /= scale;
    return *this;
}
bool energy_t::operator==(const energy_t &other) {
    bool result = abs(e1b.real() - other.e1b.real()) < 1e-12;
    result &= abs(e1b.imag() - other.e1b.imag()) < 1e-12;
    result &= abs(e2b.real() - other.e2b.real()) < 1e-12;
    result &= abs(e2b.imag() - other.e2b.imag()) < 1e-12;
    result &= abs(etot.real() - other.etot.real()) < 1e-12;
    result &= abs(etot.imag() - other.etot.imag()) < 1e-12;
    return result;
}
std::ostream &operator<<(std::ostream &os, energy_t &energy) {
    os << std::fixed << std::setprecision(4) << energy.etot << " " << energy.e1b << " " << energy.e2b;
    return os;
}
}  // namespace ipie