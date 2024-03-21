#include "determinant.h"

#include <sstream>

namespace ipie {

bool Determinant::operator==(const Determinant& other) const {
    return (alpha == other.alpha) && (beta == other.beta);
}

std::ostream& operator<<(std::ostream& os, const Determinant& det) {
    os << det.alpha << " " << det.beta;
    return os;
}

size_t Determinant::count_difference(const Determinant& other) const {
    return alpha.count_difference(other.alpha) + beta.count_difference(other.beta);
}

}  // namespace ipie