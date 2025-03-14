#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "ineurons_layer.h"

class ConvolutionLayer : public INeuronsLayer
{
public:
    ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_size, const unsigned int &stride);
    Vector<float> compute_outputs(const Vector<float> &input_vector);
    void adapt_gradient(ILayer &previous_layer, Vector<float> &dCdZ, const float &epsilon, Vector<float> &dCdZprime);
    void apply_new_weights(const float &max_gradiant);
};

#endif // CONVOLUTION_LAYER_H
