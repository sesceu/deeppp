#ifndef DEEPPP_LAYER_H
#define DEEPPP_LAYER_H

#include <Eigen/Core>

namespace deeppp
{

class Layer
{
 public:
    typedef Eigen::VectorXd InputVector;
    typedef Eigen::VectorXd OutputVector;

    /**
     * @brief Layer
     * @param input_neurons
     * @param output_neurons
     */
    Layer(std::size_t input_neurons, std::size_t output_neurons);

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
     * @brief InputNeurons
     * @return
     */
    virtual std::size_t InputNeurons() const;

    /**
     * @brief OutputNeurons
     * @return
     */
    virtual std::size_t OutputNeurons() const;

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

 private:

    //! the number of input neurons
    std::size_t input_neurons_;

    //! the number of output neurons
    std::size_t output_neurons_;

    //! the vector of inputs
    InputVector input_;

    //! the vector of outputs
    OutputVector output_;
};

}  // namespace deeppp

#endif // DEEPPP_LAYER_H
