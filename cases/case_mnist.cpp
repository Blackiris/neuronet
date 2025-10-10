#include "case_mnist.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>

#include "../neurons_network/neurons_network.h"
#include "../neurons_network/neurons_network_factory.h"
#include "../neurons_network/network_trainer.h"
#include "../neurons_network/vector.h"
#include "../std_vector_util.h"

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

#include <random>

CaseMnist::CaseMnist() {}


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

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<unsigned char> training_labels = readLabels("train-labels.idx1-ubyte");
    std::vector<Image> training_images = readImages("train-images.idx3-ubyte");
    std::vector<TrainingData> training_datas = convertToTrainingDatas(training_images, training_labels);
    std::shuffle(training_datas.begin(), training_datas.end(), g);
    std::vector<TrainingData> training_datas_small(&training_datas[0], &training_datas[50]);

    std::vector<unsigned char> test_labels = readLabels("t10k-labels.idx1-ubyte");
    std::vector<Image> test_images = readImages("t10k-images.idx3-ubyte");
    std::vector<TrainingData> test_datas = convertToTrainingDatas(test_images, test_labels);
    std::shuffle(test_datas.begin(), test_datas.end(), g);
    std::vector<TrainingData> test_datas_small(&test_datas[0], &test_datas[1000]);



#ifdef GPERFTOOLS
    ProfilerStart("mnist.prof");
#endif


    // Works
    // const unsigned int input_size = training_datas[0].input.size();
    // NeuronsNetwork* network = NeuronsNetworkFactory::create_network(input_size, 16, 10, 2);

    // NetworkTrainer network_trainer;
    // std::vector<std::vector<TrainingData>> datas_chunks = StdVectorUtil::split_chunks(training_datas, 5000);

    // network_trainer.train_network(*network, {datas_chunks[0]}, test_datas_small, {0.1, 0.0001, 1000, 0});
    // network_trainer.test_network(*network, test_datas);


    // Works
    // const unsigned int input_size = training_datas[0].input.size();
    // NeuronsNetwork* network = NeuronsNetworkFactory::create_network(input_size, 100, 10, 3);

    // NetworkTrainer network_trainer;
    // std::vector<std::vector<TrainingData>> datas_chunks = StdVectorUtil::split_chunks(training_datas, 1000);

    // network_trainer.train_network(*network, {datas_chunks[0]}, test_datas_small, {1, 0.0001, 1000, 0, 0.9, 0.99});
    // network_trainer.test_network(*network, test_datas);

    // Conv NET
    std::unique_ptr<NeuronsNetwork> network = NeuronsNetworkFactory::create_conv_network(training_images[0].nb_cols,training_images[0].nb_rows,
                                                                        10, 6, 16);

    NetworkTrainer network_trainer;
    std::vector<std::vector<TrainingData>> datas_chunks = StdVectorUtil::split_chunks(training_datas, 2000);

    network_trainer.train_network(*network, datas_chunks, test_datas_small, {1, 0.001, 20, 0});

#ifdef GPERFTOOLS
    ProfilerStop();
#endif

    network_trainer.test_network(*network, test_datas);
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
    for (unsigned int i=0; i<size; i++) {
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
