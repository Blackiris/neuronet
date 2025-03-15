#include "input_layer.h"

InputLayer::InputLayer(): ILayer(0) {}

Vector<float> InputLayer::compute_outputs(const Vector<float> &input_vector) {
    m_outputs = input_vector;
    return m_outputs;
}
