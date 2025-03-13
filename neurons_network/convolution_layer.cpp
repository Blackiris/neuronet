#include "convolution_layer.h"


ConvolutionLayer::ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_size, const unsigned int &stride)
    : NeuronsLayer(0, 0) {

}

Vector<float> ConvolutionLayer::compute_outputs(const Vector<float> &input_vector) {

    return m_outputs;
}
