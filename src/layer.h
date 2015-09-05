#ifndef LAYER_H
#define LAYER_H

#include <cstddef>
#include <math.h>
#include <Eigen/Core>

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
class Layer
{
 public:
    typedef Eigen::Matrix<FloatType, input_neurons, 1> InputVector;
    typedef Eigen::Matrix<FloatType, output_neurons, 1> OutputVector;

    Layer();

    /**
     * @brief Input
     * @param input
     */
    virtual void Input(const InputVector& input);

    /**
     * @brief Input
     * @return
     */
    virtual InputVector Input() const;

    /**
     * @brief Input
     * @return
     */
    virtual InputVector& Input();

    /**
     * @brief Output
     * @param output
     */
    virtual void Output(const OutputVector& output);

    /**
     * @brief Output
     * @return
     */
    virtual OutputVector Output() const;

    /**
     * @brief Output
     * @return
     */
    virtual OutputVector& Output();

    /**
     * @brief up calculates the outputs based on the inputs
     */
    virtual void Up() = 0;

    /**
     * @brief down calculates the inputs based on the outputs
     */
    virtual void Down() = 0;

    /**
     * @brief Train
     */
    virtual void Train() = 0;

    /**
     * @brief logistic
     * @param value
     * @return
     */
    static FloatType logistic(FloatType value);

 private:

    //! the vector of inputs
    InputVector input_;

    //! the vector of outputs
    OutputVector output_;
};

//! implementation

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
Layer<FloatType, input_neurons, output_neurons>::Layer()
{
    input_.setConstant(0);
    output_.setConstant(0);
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
FloatType Layer<FloatType, input_neurons, output_neurons>::logistic(FloatType value)
{
    return 1.0 / (1.0 + std::exp(-value));
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void Layer<FloatType, input_neurons, output_neurons>::Input(const InputVector &input)
{
    input_ = input;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename Layer<FloatType, input_neurons, output_neurons>::InputVector Layer<FloatType, input_neurons, output_neurons>::Input() const
{
    return input_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename Layer<FloatType, input_neurons, output_neurons>::InputVector& Layer<FloatType, input_neurons, output_neurons>::Input()
{
    return input_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
void Layer<FloatType, input_neurons, output_neurons>::Output(const OutputVector& output)
{
    output_ = output;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename Layer<FloatType, input_neurons, output_neurons>::OutputVector Layer<FloatType, input_neurons, output_neurons>::Output() const
{
    return output_;
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons>
typename Layer<FloatType, input_neurons, output_neurons>::OutputVector& Layer<FloatType, input_neurons, output_neurons>::Output()
{
    return output_;
}

#endif // LAYER_H
