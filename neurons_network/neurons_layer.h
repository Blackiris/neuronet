#ifndef NEURONS_LAYER_H
#define NEURONS_LAYER_H

#include "ineurons_layer.h"

#include <functional>
#include <vector>
#include <random>

class NeuronsLayer : public INeuronsLayer
{
public:
    NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights);

    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) override;
    void apply_new_weights(const float &epsilon, const float &max_gradiant) override;


    std::vector<Vector<float>> m_weights_mat;
    std::vector<Vector<float>> m_weights_mat_delta;
    std::vector<float> m_biases;
    std::vector<float> m_biases_delta;

    std::function<float(float)> m_activation_fun = [](float x) { return x < 0 ? 0 : x; };
    std::function<float(float)> m_deriv_activation_fun = [](float x) { return x < 0 ? 0 : 1; };

private:
    static std::random_device rd;
    static std::mt19937 gen;
};

#endif // NEURONS_LAYER_H
