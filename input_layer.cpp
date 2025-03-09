#include "input_layer.h"

InputLayer::InputLayer() {}

Vector<float> InputLayer::compute_outputs(const Vector<float> &input_vector) {
    m_outputs = input_vector;
    return m_outputs;
}

unsigned int InputLayer::get_output_size() {
    return m_outputs.size();
}
