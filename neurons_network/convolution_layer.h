#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "ineurons_layer.h"

class ConvolutionLayer : public INeuronsLayer
{
public:
    ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_radius);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;
    void adapt_gradient(ILayer &previous_layer, Vector<float> &dCdZ, const float &epsilon, Vector<float> &dCdZprime) override;
    void apply_new_weights(const float &max_gradiant) override;

private:
    std::vector<std::vector<float>> m_conv_weights, m_conv_weights_delta;
    unsigned int m_conv_radius;
    unsigned int m_input_x, m_input_y;
};

#endif // CONVOLUTION_LAYER_H
