#include "mnistloader.h"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace deeppp
{

MNISTLoader::ImageList MNISTLoader::Load(const std::string filename)
{
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!input_stream.is_open() || !input_stream.good() || input_stream.eof())
        throw std::runtime_error("failed to open mnist file.");

    int32_t magic_number;
    input_stream.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    SwapEndianess(magic_number);

    int32_t number_of_images;
    input_stream.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    SwapEndianess(number_of_images);

    int32_t rows;
    input_stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    SwapEndianess(rows);

    int32_t columns;
    input_stream.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    SwapEndianess(columns);

    std::cout << "mnist file contents:\n";
    std::cout << " magic number: " << magic_number << std::endl;
    std::cout << " # images:     " << number_of_images << std::endl;
    std::cout << " rows:         " << rows << std::endl;
    std::cout << " columns:      " << columns << std::endl;

    if (rows != 28)
        throw std::runtime_error("image rows is not 28.");
    if (columns != 28)
        throw std::runtime_error("image columns is not 28.");

    std::list<MNISTImage<28, 28>> image_list;
    for (int i = 0; i < number_of_images; ++i)
    {
        std::array<unsigned char, 28 * 28> image_data;
        input_stream.read(reinterpret_cast<char*>(image_data.data()), rows * columns);
        image_list.emplace_back(MNISTImage<28, 28>(image_data));
    }
    return image_list;
}

}  // namespace deeppp
