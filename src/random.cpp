#include "random.h"

namespace deeppp
{

std::random_device Random::random_device_;
std::mt19937 Random::random_engine_(Random::random_device_());
std::uniform_real_distribution<double> Random::uniform_01_(0, 1);

double Random::Uniform01()
{
    return uniform_01_(random_engine_);
}

}  // namespace deeppp
