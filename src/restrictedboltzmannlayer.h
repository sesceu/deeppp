#ifndef DEEPPP_RESTRICTEDBOLTZMANNLAYER_H
#define DEEPPP_RESTRICTEDBOLTZMANNLAYER_H

#include "layer.h"
#include "random.h"

namespace deeppp
{

class RestrictedBoltzmannLayer : public Layer
{
 public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WeightMatrix;
    typedef Layer Base;
    typedef typename Base::InputVector InputBiasVector;
    typedef typename Base::OutputVector OutputBiasVector;

    /**
     * @brief RestrictedBoltzmannLayer
     * @param input_neurons
     * @param output_neurons
     * @param learning_rate the learning rate (should be around 0.01)
     */
    RestrictedBoltzmannLayer(std::size_t input_neurons, std::size_t output_neurons, double learning_rate);

    /**
     * @brief InputProbabilities
     * @return
     */
    virtual typename Base::InputVector InputProbabilities() const;

    /**
     * @brief OutputProbabilities
     * @return
     */
    virtual typename Base::OutputVector OutputProbabilities() const;

    /**
     * @brief Weight
     * @return
     */
    virtual WeightMatrix Weight() const;

    /**
     * @brief Weight
     * @return
     */
    virtual WeightMatrix& Weight();

    /**
     * @brief InputBias
     * @return
     */
    virtual InputBiasVector InputBias() const;

    /**
     * @brief InputBias
     * @return
     */
    virtual InputBiasVector& InputBias();

    /**
     * @brief OutputBias
     * @return
     */
    virtual OutputBiasVector OutputBias() const;

    /**
     * @brief OutputBias
     * @return
     */
    virtual OutputBiasVector& OutputBias();

    void Input(const typename Base::InputVector& input) override;
    using Base::Input;

    void Output(const typename Base::OutputVector& output) override;
    using Base::Output;

    void Up() override;

    void Down() override;

    void Train() override;

 private:

    //! the vector of input probabilities
    typename Base::InputVector input_probabilities_;

    //! the vector of output probabilities
    typename Base::OutputVector output_probabilities_;

    //! the weight matrix
    WeightMatrix weight_;

    //! the input bias
    InputBiasVector input_bias_;

    //! the output bias
    OutputBiasVector output_bias_;

    //! the learning rate
    double learning_rate_;
};

}  // namespace deeppp

#endif // DEEPPP_RESTRICTEDBOLTZMANNLAYER_H
