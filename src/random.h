#ifndef DEEPPP_RANDOM_H
#define DEEPPP_RANDOM_H

#include <random>

namespace deeppp
{

class Random
{
 public:

    /**
     * @brief Uniform01
     * @return a random sample x \in [0;1]
     */
    static double Uniform01();

 private:

    //! the random device
    static std::random_device random_device_;

    //! the random engine (mersenne twister)
    static std::mt19937 random_engine_;

    //! the uniform distribution
    static std::uniform_real_distribution<double> uniform_01_;
};

}  // namespace deeppp

#endif // DEEPPP_RANDOM_H
