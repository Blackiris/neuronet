#ifndef CASE_MNIST_H
#define CASE_MNIST_H

#include "Case.h"
#include <cstdint>
#include <string>
#include <vector>

#include "../training_data.h"

class CaseMnist : public Case
{
private:
    struct Image {
        std::uint32_t nb_rows;
        std::uint32_t nb_cols;
        std::vector<unsigned char> pixels;
    };

public:
    CaseMnist();
    void run() override;

private:
    std::vector<unsigned char> readLabels(std::string path);
    std::vector<Image> readImages(std::string path);
    Image readImage(const std::uint32_t &nb_rows, const std::uint32_t &nb_cols, std::ifstream &inputFile, char buffer[]);
    TrainingData convertImageToTrainingData(const Image &image, const unsigned char &label);

};

#endif // CASE_MNIST_H
