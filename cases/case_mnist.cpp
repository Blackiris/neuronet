#include "case_mnist.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

CaseMnist::CaseMnist() {}

void CaseMnist::run() {
    std::vector<unsigned char> labels = readLabels("train-labels.idx1-ubyte");
    std::vector<Image> images = readImages("train-images.idx3-ubyte");
}

std::uint32_t read32bits(char* buffer, const int &pos) {
    return (reinterpret_cast<unsigned char&>(buffer[pos]) << 24)
    | (reinterpret_cast<unsigned char&>(buffer[pos+1]) << 16)
        | (reinterpret_cast<unsigned char&>(buffer[pos+2]) << 8)
        | reinterpret_cast<unsigned char&>(buffer[pos+3]);
}

std::vector<unsigned char> CaseMnist::readLabels(std::string path) {
    std::ifstream inputFile(path, std::ios::binary);

    if (!inputFile) {
        std::cerr << "Erreur d'ouverture du fichier." << std::endl;
    }

    const std::size_t bufferSize = 10;
    char buffer[bufferSize];
    std::streamsize bytesRead;

    inputFile.read(buffer, 8);
    const std::uint32_t size_labels = read32bits(buffer, 4);
    std::vector<unsigned char> labels;
    labels.reserve(size_labels);

    do {
        inputFile.read(buffer, bufferSize);

        bytesRead = inputFile.gcount();
        for (std::streamsize i = 0; i < bytesRead; ++i) {
            labels.push_back(reinterpret_cast<unsigned char&>(buffer[i]));
        }
    } while(bytesRead > 0);

    inputFile.close();
    return labels;
}

std::vector<CaseMnist::Image> CaseMnist::readImages(std::string path) {
    std::ifstream inputFile(path, std::ios::binary);

    if (!inputFile) {
        std::cerr << "Erreur d'ouverture du fichier." << std::endl;
    }

    char header_buffer[16];

    inputFile.read(header_buffer, 16);
    const std::uint32_t nb_images = read32bits(header_buffer, 4);
    const std::uint32_t nb_rows = read32bits(header_buffer, 8);
    const std::uint32_t nb_cols = read32bits(header_buffer, 12);

    const std::uint32_t nb_pixels_per_image = nb_rows*nb_cols;
    auto image_buffer = std::make_unique<char[]>(nb_pixels_per_image);

    std::vector<Image> images;
    images.reserve(nb_images);

    for (std::uint32_t i=0; i<nb_images; i++) {
        images.push_back(readImage(nb_rows, nb_cols, inputFile, image_buffer.get()));
    }

    inputFile.close();
    return images;
}

CaseMnist::Image CaseMnist::readImage(const std::uint32_t &nb_rows, const std::uint32_t &nb_cols, std::ifstream &inputFile, char buffer[]) {
    const std::uint32_t nb_pixels_per_image = nb_rows*nb_cols;
    std::vector<unsigned char> colors;
    colors.reserve(nb_pixels_per_image);


    inputFile.read(buffer, nb_pixels_per_image);
    for (std::uint32_t i=0; i<nb_pixels_per_image; i++) {
        colors.push_back(buffer[i]);
    }
    return Image(nb_rows, nb_cols, colors);
}
