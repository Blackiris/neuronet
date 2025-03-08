#include "neurons_network.h"

#include "neurons_layer.h"

NeuronsNetwork::NeuronsNetwork() {

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
