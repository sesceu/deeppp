#ifndef TIMESERIESLAYER_H
#define TIMESERIESLAYER_H

#include "layer.h"

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
class TimeseriesLayer : public LayerType<FloatType, input_neurons * series_length, output_neurons>
{
 public:
    typedef Eigen::Matrix<FloatType, input_neurons, 1> SeriesElementVector;
    typedef LayerType<FloatType, input_neurons * series_length, output_neurons> Base;

    /**
     * @brief TimeseriesLayer
     * @param learning_rate
     */
    TimeseriesLayer(FloatType learning_rate);

    void Input(const typename Base::InputVector& input) override;
    using Base::Input;

    /**
     * @brief Append adds a new series element (moves existing elements)
     * @param series_element the element to be added
     * @return true if enough elements have been added, false otherwise
     */
    virtual bool Append(const SeriesElementVector& series_element);

    /**
     * @brief ResetTimeseries clears the time series elements (appending starts again)
     */
    virtual void ResetTimeseries();

    /**
     * @brief ReplaceLatest replaces the latest series element
     * @param series_element the element to replace with
     */
    virtual void ReplaceLatest(const SeriesElementVector& series_element);

 private:

    //! the number of added elements, in order to decide when the input vector is complete
    std::size_t number_of_added_elements_;

    //! the vector that contains the added elements (is not modified by Up/Down/Train)
    typename Base::InputVector time_series_vector_;

};

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
TimeseriesLayer<FloatType, input_neurons, output_neurons, series_length, LayerType>::TimeseriesLayer(FloatType learning_rate)
    : LayerType<FloatType, input_neurons * series_length, output_neurons>(learning_rate),
      number_of_added_elements_(0)
{
    ResetTimeseries();
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
void TimeseriesLayer<FloatType, input_neurons, output_neurons, series_length, LayerType>::Input(const typename Base::InputVector& input)
{
    time_series_vector_ = input;
    Base::Input(time_series_vector_);
}


template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
bool TimeseriesLayer<FloatType, input_neurons, output_neurons, series_length, LayerType>::Append(const SeriesElementVector& series_element)
{
    time_series_vector_.head(input_neurons * (series_length - 1)) =
        time_series_vector_.tail(input_neurons * (series_length - 1));
    time_series_vector_.tail(input_neurons) = series_element;
    this->Input(time_series_vector_);
    number_of_added_elements_++;

    if (number_of_added_elements_ >= series_length)
    {
        number_of_added_elements_ = series_length;
        return true;
    }
    else
    {
        return false;
    }
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
void TimeseriesLayer<FloatType, input_neurons, output_neurons, series_length, LayerType>::ResetTimeseries()
{
    number_of_added_elements_ = 0;
    time_series_vector_.setConstant(0.5);
}

template <typename FloatType, std::size_t input_neurons, std::size_t output_neurons, std::size_t series_length, template <typename, std::size_t, std::size_t> class LayerType>
void TimeseriesLayer<FloatType, input_neurons, output_neurons, series_length, LayerType>::ReplaceLatest(const SeriesElementVector& series_element)
{
    time_series_vector_.tail(input_neurons) = series_element;
    this->Input(time_series_vector_);
}

#endif // TIMESERIESLAYER_H
