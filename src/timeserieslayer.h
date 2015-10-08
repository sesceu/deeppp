#ifndef DEEPPP_TIMESERIESLAYER_H
#define DEEPPP_TIMESERIESLAYER_H

#include "layer.h"

namespace deeppp
{

template <typename LayerType>
class TimeseriesLayer : public LayerType
{
 public:
    typedef Eigen::VectorXd SeriesElementVector;
    typedef LayerType Base;

    /**
     * @brief TimeseriesLayer
     * @param learning_rate
     */
    TimeseriesLayer(std::size_t input_neurons, std::size_t output_neurons, std::size_t time_series, double learning_rate);

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

    //! the number of time series
    std::size_t time_series_;

    //! the length of one time series
    std::size_t time_series_length_;

    //! the number of added elements, in order to decide when the input vector is complete
    std::size_t number_of_added_elements_;

    //! the vector that contains the added elements (is not modified by Up/Down/Train)
    typename Base::InputVector time_series_vector_;

};

template <typename LayerType>
TimeseriesLayer<LayerType>::TimeseriesLayer(std::size_t input_neurons, std::size_t output_neurons, std::size_t time_series, double learning_rate)
    : LayerType(input_neurons * time_series, output_neurons, learning_rate),
      time_series_(time_series), time_series_length_(input_neurons),
      number_of_added_elements_(0), time_series_vector_(input_neurons * time_series)
{
    ResetTimeseries();
}

template <typename LayerType>
void TimeseriesLayer<LayerType>::Input(const typename Base::InputVector& input)
{
    time_series_vector_ = input;
    Base::Input(time_series_vector_);
}


template <typename LayerType>
bool TimeseriesLayer<LayerType>::Append(const SeriesElementVector& series_element)
{
    assert(static_cast<std::size_t>(series_element.rows()) == time_series_length_);
    time_series_vector_.head(time_series_length_ * (time_series_ - 1)) =
        time_series_vector_.tail(time_series_length_ * (time_series_ - 1));
    time_series_vector_.tail(time_series_length_) = series_element;
    this->Input(time_series_vector_);
    number_of_added_elements_++;

    if (number_of_added_elements_ >= time_series_)
    {
        number_of_added_elements_ = time_series_;
        return true;
    }
    else
    {
        return false;
    }
}

template <typename LayerType>
void TimeseriesLayer<LayerType>::ResetTimeseries()
{
    number_of_added_elements_ = 0;
    time_series_vector_.setConstant(0.5);
    Input(time_series_vector_);
}

template <typename LayerType>
void TimeseriesLayer<LayerType>::ReplaceLatest(const SeriesElementVector& series_element)
{
    assert(static_cast<std::size_t>(series_element.rows()) == time_series_length_);
    time_series_vector_.tail(time_series_length_) = series_element;
    Input(time_series_vector_);
}

}  // namespace deeppp

#endif // DEEPPP_TIMESERIESLAYER_H
