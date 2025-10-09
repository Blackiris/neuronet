#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "ineurons_layer.h"
#include <functional>
#include <random>

class ConvolutionLayer : public INeuronsLayer
{
public:
    ConvolutionLayer(const unsigned &input_x, const unsigned &input_y, const unsigned &conv_radius, std::vector<unsigned> links_table);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;
    void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned &dcdz_prime_offset) override;
    void apply_new_weights(const TrainingParams &training_params) override;

private:
    std::vector<unsigned int> m_links_table;
    std::vector<Vector<float>> m_conv_weights, m_conv_weights_delta, m_weights_momentum;
    Vector<float> variance_moment;
    unsigned int m_conv_radius;
    unsigned int m_conv_diameter;
    unsigned int m_input_x, m_input_y;
    Vector<float> biases, biases_delta;
    unsigned int m_output_x;

    std::function<float(float)> m_activation_fun = [](float x) { return x < 0 ? 0 : x; };
    std::function<float(float)> m_deriv_activation_fun = [](float x) { return x < 0 ? 0 : 1; };

private:
    static std::random_device rd;
    static std::mt19937 gen;
};

#endif // CONVOLUTION_LAYER_H
