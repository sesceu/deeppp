#ifndef DEEPPP_MNISTIMAGE_H
#define DEEPPP_MNISTIMAGE_H

#include <array>
#include <cstddef>
#include <stdexcept>

namespace deeppp
{

template <std::size_t rows, std::size_t columns>
class MNISTImage
{
 public:
    /**
     * @brief Size
     * @return
     */
    static constexpr std::size_t Size();

    /**
     * @brief MNISTImage
     */
    MNISTImage();

    /**
     * @brief MNISTImage
     * @param data
     */
    MNISTImage(const std::array<unsigned char, Size()>& data);

    /**
     * @brief operator []
     * @param pixel
     * @return
     */
    unsigned char& operator[] (std::size_t pixel);

    /**
     * @brief operator []
     * @param pixel
     * @return
     */
    unsigned char operator[] (std::size_t pixel) const;

 private:

    //! the image data
    std::array<unsigned char, Size()> data_;
};

//! implementation


template <std::size_t rows, std::size_t columns>
MNISTImage<rows, columns>::MNISTImage()
{}

template <std::size_t rows, std::size_t columns>
MNISTImage<rows, columns>::MNISTImage(const std::array<unsigned char, Size()>& data)
    : data_(data)
{}

template <std::size_t rows, std::size_t columns>
constexpr std::size_t MNISTImage<rows, columns>::Size()
{
    return rows * columns;
}

template <std::size_t rows, std::size_t columns>
unsigned char& MNISTImage<rows, columns>::operator[] (std::size_t pixel)
{
    if (pixel < 0 || pixel >= Size())
        throw std::runtime_error("pixel index out of range.");
    return data_[pixel];
}

template <std::size_t rows, std::size_t columns>
unsigned char MNISTImage<rows, columns>::operator[] (std::size_t pixel) const
{
    if (pixel < 0 || pixel >= Size())
        throw std::runtime_error("pixel index out of range.");
    return data_[pixel];
}

}  // namespace deeppp

#endif // MNISTIMAGE_H
