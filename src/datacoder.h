#ifndef DATACODER_H
#define DATACODER_H

#include <Eigen/Core>

template <typename Source, typename Destination>
class DataCoder
{
 public:
    /**
     * @brief encode
     * @param data
     * @return
     */
    static Destination encode(const Source& data);

    /**
     * @brief decode
     * @param vector
     * @return
     */
    static Source decode(const Destination& vector);
};

//! implementation

template <typename Source, typename Destination>
Destination DataCoder<Source, Destination>::encode(const Source& data)
{
    return data;
}

template <typename Source, typename Destination>
Source DataCoder<Source, Destination>::decode(const Destination& data)
{
    return data;
}

#endif // DATACODER_H
