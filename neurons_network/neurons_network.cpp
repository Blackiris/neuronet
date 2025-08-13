#include "neurons_network.h"


NeuronsNetwork::NeuronsNetwork() noexcept {}

Vector<float> NeuronsNetwork::compute(const Vector<float> &input) {
    Vector<float> intermediate_input = m_input_layer.compute_outputs(input);

    for (auto&& layer: m_layers) {
        intermediate_input = layer->compute_outputs(intermediate_input);
    }

    return intermediate_input;
}


void NeuronsNetwork::apply_new_weights(const TrainingParams &training_params) {
    for (auto&& layer: m_layers) {
        layer->apply_new_weights(training_params);
    }
}
