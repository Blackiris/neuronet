#include "convolution_layer.h"


ConvolutionLayer::ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_size, const unsigned int &stride)
    : INeuronsLayer() {

}

Vector<float> ConvolutionLayer::compute_outputs(const Vector<float> &input_vector) {

    return m_outputs;
}

void ConvolutionLayer::adapt_gradient(ILayer &previous_layer, Vector<float> &dCdZ, const float &epsilon, Vector<float> &dCdZprime) {

}

void ConvolutionLayer::apply_new_weights(const float &max_gradiant) {

}
