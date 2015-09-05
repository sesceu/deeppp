#ifndef RESTRICTEDBOLTZMANNLAYER_H
#define RESTRICTEDBOLTZMANNLAYER_H

#include "layer.h"
#include "random.h"

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
class RestrictedBoltzmannLayer : public Layer<FloatType, input_neurons, output_neurons>
{
 public:
    typedef Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> WeightMatrix;
    typedef Layer<FloatType, input_neurons, output_neurons> Base;
    typedef typename Base::InputVector InputBiasVector;
    typedef typename Base::OutputVector OutputBiasVector;

    /**
     * @brief RestrictedBoltzmannLayer
     * @param epsilon the learning rate (should be around 0.01)
     */
    RestrictedBoltzmannLayer(FloatType epsilon);

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
    FloatType epsilon_;
};

//! implementation

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::RestrictedBoltzmannLayer(FloatType epsilon)
    : Layer<FloatType, input_neurons, output_neurons>(), epsilon_(epsilon)
{
    input_probabilities_.setConstant(0);
    output_probabilities_.setConstant(0);
    FloatType maximum_random_weight = 1.0 / input_neurons;
    auto uniform = [&] (FloatType) {return static_cast<FloatType>(Random::Uniform01() * 2.0 * maximum_random_weight - maximum_random_weight);};
    weight_ = WeightMatrix::NullaryExpr(output_neurons, input_neurons, uniform);
    input_bias_.setConstant(0);
    output_bias_.setConstant(0);
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Base::InputVector RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::InputProbabilities() const
{
    return input_probabilities_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Base::OutputVector RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::OutputProbabilities() const
{
    return output_probabilities_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::WeightMatrix RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Weight() const
{
    return weight_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::WeightMatrix& RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Weight()
{
    return weight_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::InputBiasVector RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::InputBias() const
{
    return input_bias_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::InputBiasVector& RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::InputBias()
{
    return input_bias_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::OutputBiasVector RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::OutputBias() const
{
    return output_bias_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::OutputBiasVector& RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::OutputBias()
{
    return output_bias_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Input(const typename Base::InputVector& input)
{
    Layer<FloatType, input_neurons, output_neurons>::Input(input);
    input_probabilities_ = input;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Output(const typename Base::OutputVector& output)
{
    Layer<FloatType, input_neurons, output_neurons>::Output(output);
    output_probabilities_ = output;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Up()
{
    output_probabilities_ = weight_ * this->Input() + output_bias_;
    for (std::size_t i = 0; i < output_neurons; ++i)
    {
        output_probabilities_(i) = this->logistic(output_probabilities_(i));
        if (Random::Uniform01() < output_probabilities_(i))
            Output()(i) = 1.0;
        else
            Output()(i) = 0.0;
    }
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Down()
{
    input_probabilities_ = weight_.transpose() * Output() + input_bias_;
    for (std::size_t i = 0; i < input_neurons; ++i)
    {
        input_probabilities_(i) = this->logistic(input_probabilities_(i));
        if (Random::Uniform01() < input_probabilities_(i))
            Input()(i) = 1.0;
        else
            Input()(i) = 0.0;
    }
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void RestrictedBoltzmannLayer<FloatType, input_neurons, output_neurons>::Train()
{
    typename Base::InputVector initial_input(Input());
    Up();

    WeightMatrix data_correlation(output_neurons, input_neurons);
    for (std::size_t output_index = 0; output_index < output_neurons; ++output_index)
    {
        for (std::size_t input_index = 0; input_index < input_neurons; ++input_index)
        {
            data_correlation(output_index, input_index) = output_probabilities_(output_index) * Input()(input_index);
        }
    }

    typename Base::OutputVector initial_output_probabilities(OutputProbabilities());

    Down();
    Up();

    WeightMatrix reconstructed_correlation(output_neurons, input_neurons);
    for (std::size_t output_index = 0; output_index < output_neurons; ++output_index)
    {
        for (std::size_t input_index = 0; input_index < input_neurons; ++input_index)
        {
            reconstructed_correlation(output_index, input_index) = output_probabilities_(output_index) * Input()(input_index);
        }
    }

    weight_ += (data_correlation - reconstructed_correlation) * epsilon_;
    output_bias_ += (initial_output_probabilities - OutputProbabilities()) * epsilon_;
    input_bias_ += (initial_input - Input()) * epsilon_;
}

#endif // RESTRICTEDBOLTZMANNLAYER_H
