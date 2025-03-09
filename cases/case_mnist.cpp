#include "case_mnist.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include "../neurons_network.h"
#include "../neuronsnetworkfactory.h"
#include "../network_trainer.h"
#include "../vector.h"
#include "../vector_util.h"

CaseMnist::CaseMnist() {}


unsigned int mapNetworkOutputToRes(const Vector<float> &output) {
    return VectorUtil::find_max_pos<float>(output);
}

bool isResultGood(const Vector<float> &expected, const Vector<float> &actual) {
    return mapNetworkOutputToRes(expected) == mapNetworkOutputToRes(actual);
}

void testNetwork(std::vector<TrainingData> &datas, NeuronsNetwork* network) {
    unsigned int correct = 0;

    for (unsigned int i=0; i<datas.size(); i++) {
        const TrainingData data = datas[i];
        Vector<float> actual_res = network->compute(data.input);
        if (isResultGood(data.res, actual_res)) {
            correct++;
        }
    }
    std::cout << correct << "/" << datas.size() << "\n";
}



std::vector<float> mapIntToNetworkOuput(const unsigned char i) {
    std::vector<float> res;
    res.assign(10, 0);
    res[i] = 1;
    return res;
}


std::uint32_t read32bits(char* buffer, const int &pos) {
    return (reinterpret_cast<unsigned char&>(buffer[pos]) << 24)
    | (reinterpret_cast<unsigned char&>(buffer[pos+1]) << 16)
        | (reinterpret_cast<unsigned char&>(buffer[pos+2]) << 8)
        | reinterpret_cast<unsigned char&>(buffer[pos+3]);
}

void CaseMnist::run() {
    std::vector<unsigned char> training_labels = readLabels("train-labels.idx1-ubyte");
    std::vector<Image> training_images = readImages("train-images.idx3-ubyte");
    std::vector<TrainingData> training_datas = convertToTrainingDatas(training_images, training_labels);
    std::vector<TrainingData> training_datas_small(&training_datas[0], &training_datas[10]);

    std::vector<unsigned char> test_labels = readLabels("t10k-labels.idx1-ubyte");
    std::vector<Image> test_images = readImages("t10k-images.idx3-ubyte");
    std::vector<TrainingData> test_datas = convertToTrainingDatas(test_images, test_labels);

    const unsigned int input_size = training_datas[0].input.size();

    NeuronsNetwork* network = NeuronsNetworkFactory::createNetwork(input_size, 16, 10, 2);
    NetworkTrainer network_trainer;
    network_trainer.train_network(*network, training_datas_small, 0.00001, 5000);

    testNetwork(test_datas, network);
}


TrainingData CaseMnist::convertImageToTrainingData(const Image &image, const unsigned char &label) {
    std::function<float(unsigned char)> unary_op = [](unsigned char num) {
        return num/255.f;
    };
    std::vector<float> pixels;
    std::transform(image.pixels.begin(), image.pixels.end(), std::back_inserter(pixels), unary_op);
    return TrainingData(pixels, mapIntToNetworkOuput(label));
}

std::vector<TrainingData> CaseMnist::convertToTrainingDatas(const std::vector<Image> images, const std::vector<unsigned char> &labels) {
    size_t size = labels.size();
    std::vector<TrainingData> training_datas;
    training_datas.reserve(size);
    for (int i=0; i<size; i++) {
        training_datas.push_back(convertImageToTrainingData(images[i], labels[i]));
    }
    return training_datas;
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
