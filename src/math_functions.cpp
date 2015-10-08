#include <cmath>
#include "./math_functions.h"

namespace deeppp
{

double Math::Logistic(double value)
{
    return 1.0 / (1.0 + std::exp(-value));
}

}  // namespace deeppp
