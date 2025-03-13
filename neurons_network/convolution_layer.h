#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include "neurons_layer.h"

class ConvolutionLayer : public NeuronsLayer
{
public:
    ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_size, const unsigned int &stride);
    Vector<float> compute_outputs(const Vector<float> &input_vector);
};

#endif // CONVOLUTION_LAYER_H
