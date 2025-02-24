#include "neurons_network.h"

#include <iostream>
#include "neurons_layer.h"

NeuronsNetwork::NeuronsNetwork() {
    std::unique_ptr<NeuronsLayer> neurons_layer(new NeuronsLayer(10, 10));
    std::unique_ptr<NeuronsLayer> neurons_layer2(new NeuronsLayer(10, 10));
    m_layers.push_back(std::move(neurons_layer));
    //m_layers.push_back(std::move(neurons_layer2));
}

Vector<float> NeuronsNetwork::compute(const Vector<float> &input) {
    Vector<float> intermediate_input = m_input_layer.compute_outputs(input);
    Vector<float> intermediate_output;

    for (auto&& layer: m_layers) {
        intermediate_output = layer->compute_outputs(intermediate_input);
        intermediate_input = intermediate_output;
    }

    return intermediate_output;
}
