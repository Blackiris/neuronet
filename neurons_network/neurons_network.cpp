#include "neurons_network.h"


NeuronsNetwork::NeuronsNetwork() noexcept {}

std::vector<Vector<float>> NeuronsNetwork::compute(const Vector<float> &input) {
    std::vector<Vector<float>> outputs_layers;
    outputs_layers.reserve(m_layers.size()+1);

    outputs_layers.emplace_back(input);

    for (auto&& layer: m_layers) {
        outputs_layers.emplace_back(layer->compute_outputs(outputs_layers.back()));
    }

    return outputs_layers;
}


void NeuronsNetwork::apply_new_weights(const TrainingParams &training_params) {
    for (auto&& layer: m_layers) {
        layer->apply_new_weights(training_params);
    }
}
