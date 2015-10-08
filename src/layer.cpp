#include "./layer.h"

namespace deeppp
{

Layer::Layer(std::size_t input_neurons, std::size_t output_neurons)
    : input_neurons_(input_neurons), output_neurons_(output_neurons),
      input_(input_neurons), output_(output_neurons)
{
    input_.setConstant(0);
    output_.setConstant(0);
}

void Layer::Input(const InputVector &input)
{
    assert(input.rows() == input_.rows());
    input_ = input;
}

Layer::InputVector Layer::Input() const
{
    return input_;
}

Layer::InputVector& Layer::Input()
{
    return input_;
}

Layer::OutputVector Layer::Output() const
{
    return output_;
}

Layer::OutputVector& Layer::Output()
{
    return output_;
}

void Layer::Output(const OutputVector& output)
{
    assert(output.rows() == output_.rows());
    output_ = output;
}

std::size_t Layer::InputNeurons() const
{
    return input_neurons_;
}

std::size_t Layer::OutputNeurons() const
{
    return output_neurons_;
}

}  // namespace deeppp
