#include "neurons_layer.h"

NeuronsLayer::NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights): ILayer(size) {
    for (unsigned int i=0; i<size; i++) {
        Neuron neuron(nb_weights);
        m_neurons.emplace_back(neuron);
    }
}

Vector<float> NeuronsLayer::compute_outputs(const Vector<float> &input_vector) {
    for (unsigned int i=0; i<m_neurons.size(); i++) {
        m_outputs[i] = m_neurons[i].compute_output(input_vector);
    }
    return m_outputs;
}

unsigned int NeuronsLayer::get_output_size() {
    return m_neurons.size();
}

