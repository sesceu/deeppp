#include "./math_functions.h"
#include "./restrictedboltzmannlayer.h"

namespace deeppp
{

RestrictedBoltzmannLayer::RestrictedBoltzmannLayer(std::size_t input_neurons, std::size_t output_neurons, double learning_rate)
    : Layer(input_neurons, output_neurons), input_probabilities_(input_neurons),
      output_probabilities_(output_neurons), input_bias_(input_neurons),
      output_bias_(output_neurons), learning_rate_(learning_rate)
{
    input_probabilities_.setConstant(0);
    output_probabilities_.setConstant(0);
    double maximum_random_weight = 1.0 / input_neurons;
    auto uniform = [&] (double) {return Random::Uniform01() * 2.0 * maximum_random_weight - maximum_random_weight;};
    weight_ = WeightMatrix::NullaryExpr(output_neurons, input_neurons, uniform);
    input_bias_.setConstant(0);
    output_bias_.setConstant(0);
}

RestrictedBoltzmannLayer::InputVector RestrictedBoltzmannLayer::InputProbabilities() const
{
    return input_probabilities_;
}

RestrictedBoltzmannLayer::OutputVector RestrictedBoltzmannLayer::OutputProbabilities() const
{
    return output_probabilities_;
}

RestrictedBoltzmannLayer::WeightMatrix RestrictedBoltzmannLayer::Weight() const
{
    return weight_;
}

RestrictedBoltzmannLayer::WeightMatrix& RestrictedBoltzmannLayer::Weight()
{
    return weight_;
}

RestrictedBoltzmannLayer::InputBiasVector RestrictedBoltzmannLayer::InputBias() const
{
    return input_bias_;
}

RestrictedBoltzmannLayer::InputBiasVector& RestrictedBoltzmannLayer::InputBias()
{
    return input_bias_;
}

RestrictedBoltzmannLayer::OutputBiasVector RestrictedBoltzmannLayer::OutputBias() const
{
    return output_bias_;
}

RestrictedBoltzmannLayer::OutputBiasVector& RestrictedBoltzmannLayer::OutputBias()
{
    return output_bias_;
}

void RestrictedBoltzmannLayer::Input(const Base::InputVector& input)
{
    Layer::Input(input);
    input_probabilities_ = input;
}

void RestrictedBoltzmannLayer::Output(const Base::OutputVector& output)
{
    Layer::Output(output);
    output_probabilities_ = output;
}

void RestrictedBoltzmannLayer::Up()
{
    output_probabilities_ = weight_ * Input() + output_bias_;
    for (std::size_t i = 0; i < OutputNeurons(); ++i)
    {
        output_probabilities_(i) = Math::Logistic(output_probabilities_(i));
        if (Random::Uniform01() < output_probabilities_(i))
            Output()(i) = 1.0;
        else
            Output()(i) = 0.0;
    }
}

void RestrictedBoltzmannLayer::Down()
{
    input_probabilities_ = weight_.transpose() * Output() + input_bias_;
    for (std::size_t i = 0; i < InputNeurons(); ++i)
    {
        input_probabilities_(i) = Math::Logistic(input_probabilities_(i));
        if (Random::Uniform01() < input_probabilities_(i))
            Input()(i) = 1.0;
        else
            Input()(i) = 0.0;
    }
}

void RestrictedBoltzmannLayer::Train()
{
    Base::InputVector initial_input(Input());
    Up();

    WeightMatrix data_correlation(OutputNeurons(), InputNeurons());
    for (std::size_t output_index = 0; output_index < OutputNeurons(); ++output_index)
    {
        for (std::size_t input_index = 0; input_index < InputNeurons(); ++input_index)
        {
            data_correlation(output_index, input_index) = output_probabilities_(output_index) * Input()(input_index);
        }
    }

    Base::OutputVector initial_output_probabilities(OutputProbabilities());

    Down();
    Up();

    WeightMatrix reconstructed_correlation(OutputNeurons(), InputNeurons());
    for (std::size_t output_index = 0; output_index < OutputNeurons(); ++output_index)
    {
        for (std::size_t input_index = 0; input_index < InputNeurons(); ++input_index)
        {
            reconstructed_correlation(output_index, input_index) = output_probabilities_(output_index) * Input()(input_index);
        }
    }

    weight_ += (data_correlation - reconstructed_correlation) * learning_rate_;
    output_bias_ += (initial_output_probabilities - OutputProbabilities()) * learning_rate_;
    input_bias_ += (initial_input - Input()) * learning_rate_;
}

}  // namespace deeppp
