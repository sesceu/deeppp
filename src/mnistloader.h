#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include "mnistimage.h"

#include <list>
#include <string>

class MNISTLoader
{
public:
    typedef MNISTImage<28, 28> ImageType;
    typedef std::list<ImageType> ImageList;

    static ImageList Load(const std::string filename);

    template <typename T>
    static void SwapEndianess(T&);
};

//! implementation

template <typename T>
void MNISTLoader::SwapEndianess(T& value)
{
    T swapped_value = 0;
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t byte = 0; byte < sizeof(T); ++byte)
    {
        swapped_value |= bytes[byte] << (sizeof(T) - 1 - byte) * 8;
    }
    value = swapped_value;
}

#endif // MNISTLOADER_H
