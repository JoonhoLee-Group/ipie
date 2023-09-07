#include "custom_types.h"

#include <iomanip>
#include <iostream>
#include <ostream>

namespace ipie {
new_energy_t &new_energy_t::operator+=(const new_energy_t &other) {
    etot += other.etot;
    e1b += other.e1b;
    e2b += other.e2b;
    return *this;
}
new_energy_t &new_energy_t::operator*=(const double &scale) {
    etot *= ipie::complex_t{scale};
    e1b *= ipie::complex_t{scale};
    e2b *= ipie::complex_t{scale};
    return *this;
}
new_energy_t &new_energy_t::operator*=(const ipie::complex_t &scale) {
    etot *= scale;
    e1b *= scale;
    e2b *= scale;
    return *this;
}
std::ostream &operator<<(std::ostream &os, new_energy_t &energy) {
    os << std::fixed << std::setprecision(4) << energy.etot << " " << energy.e1b << " " << energy.e2b;
    return os;
}
}  // namespace ipie