#include "neuron.h"

Neuron::Neuron(int nb_weight) : m_weights(nb_weight), m_new_weights_delta(nb_weight, 0) {
    for (int j=0; j<m_weights.size(); j++) {
        float r = (static_cast<float>(rand()) * 2 / static_cast<float>(RAND_MAX)) - 1;
        m_weights[j] = r;
    }
    m_weights.normalize();
}

Neuron::Neuron(Vector<float> &weights) :
    m_weights(weights) {}

Neuron::Neuron(Vector<float> &weights, std::function<float(float)> &lambda) :
    m_weights(weights), m_activation_fun(lambda) {}

void Neuron::apply_weight_delta(const float &max_gradiant) {
    float length = m_new_weights_delta.length();
    if (length > max_gradiant) {
        m_new_weights_delta /= length/max_gradiant;
    }
    m_weights += m_new_weights_delta;
    m_new_weights_delta.assign(0);
}

float Neuron::compute_output(Vector<float> input_vector) {
    float s = m_weights.dot(input_vector);
    m_output = m_activation_fun(s);
    return m_output;
}

float Neuron::get_output() const {
    return m_output;
}
