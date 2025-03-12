#include "neuron.h"

std::random_device Neuron::rd;
std::mt19937 Neuron::gen(Neuron::rd());

Neuron::Neuron() {}

Neuron::Neuron(int nb_weight) : m_weights(nb_weight), m_new_weights_delta(nb_weight, 0) {
    // Xavier - He init
    std::normal_distribution d{0.0, 2.0/nb_weight};

    for (int j=0; j<m_weights.size(); j++) {
        float r = d(gen);
        m_weights[j] = r;
    }
}

Neuron::Neuron(Vector<float> &weights) :
    m_weights(weights) {}

Neuron::Neuron(Vector<float> &weights, std::function<float(float)> &lambda) :
    m_activation_fun(lambda), m_weights(weights) {}

void Neuron::add_weight_delta(const unsigned int &idx, const float &delta_to_add) {
    m_new_weights_delta[idx] += delta_to_add;
}

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

float& Neuron::get_weight(const unsigned int &idx) {
    return m_weights[idx];
}
