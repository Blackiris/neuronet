#include "neurons_layer.h"

NeuronsLayer::NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights): INeuronsLayer(size) {
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

void NeuronsLayer::adapt_gradient(ILayer &previous_layer, Vector<float> &dCdZ, const float &epsilon, Vector<float> &dCdZprime) {
    for (unsigned int k=0; k<m_neurons.size(); k++) {
        Neuron &neuron = m_neurons[k];
        neuron.adapt_gradient(previous_layer, dCdZ[k], epsilon, dCdZprime);
        //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
    }
}

void NeuronsLayer::apply_new_weights(const float &max_gradiant) {
    for (auto& neuron : m_neurons) {
        neuron.apply_gradient_delta(max_gradiant);
    }
}
