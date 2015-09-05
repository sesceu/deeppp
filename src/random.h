#ifndef RANDOM_H
#define RANDOM_H

#include <random>

class Random
{
 public:

    static double Uniform01();

 private:

    //! the random device
    static std::random_device random_device_;

    //! the random engine (mersenne twister)
    static std::mt19937 random_engine_;

    //! the uniform distribution
    static std::uniform_real_distribution<double> uniform_01_;
};

#endif // RANDOM_H
