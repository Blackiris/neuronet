#include "neurons_layer.h"

NeuronsLayer::NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights): INeuronsLayer(size) {
    for (unsigned int i=0; i<size; i++) {
        Neuron neuron(nb_weights, 0.f);
        m_neurons.emplace_back(neuron);
    }
}

Vector<float> NeuronsLayer::compute_outputs(const Vector<float> &input_vector) {
    for (unsigned int i=0; i<m_neurons.size(); i++) {
        m_outputs[i] = m_neurons[i].compute_output(input_vector);
    }
    return m_outputs;
}

Vector<float> NeuronsLayer::adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ) {
    Vector<float> dCdZprime(previous_layer_output.size(), 0.f);
    for (unsigned int k=0; k<m_neurons.size(); k++) {
        Neuron &neuron = m_neurons[k];
        neuron.adapt_gradient(previous_layer_output, dCdZ[k], dCdZprime);
        //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
    }
    return dCdZprime;
    //return dCdZprime/get_output_size();
}

void NeuronsLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (auto& neuron : m_neurons) {
        neuron.apply_gradient_delta(epsilon, max_gradiant);
    }
}
