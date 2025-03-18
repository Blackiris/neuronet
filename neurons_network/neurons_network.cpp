#include "neurons_network.h"


NeuronsNetwork::NeuronsNetwork() {}

Vector<float> NeuronsNetwork::compute(const Vector<float> &input) {
    Vector<float> intermediate_input = m_input_layer.compute_outputs(input);
    Vector<float> intermediate_output;

    for (auto&& layer: m_layers) {
        intermediate_output = layer->compute_outputs(intermediate_input);
        intermediate_input = intermediate_output;
    }

    return intermediate_output;
}


void NeuronsNetwork::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (auto&& layer: m_layers) {
        layer->apply_new_weights(epsilon, max_gradiant);
    }
}
