#include "neuron.h"

std::random_device Neuron::rd;
std::mt19937 Neuron::gen(Neuron::rd());

Neuron::Neuron() {}

Neuron::Neuron(int nb_weight) : m_weights(nb_weight, 0.f), m_new_weights_delta(nb_weight, 0) {
    // Xavier - He init
    std::normal_distribution d{0.0, std::sqrt(2.0/nb_weight)};

    for (unsigned int j=0; j<m_weights.size(); j++) {
        float r = d(gen);
        m_weights[j] = r;
    }
}

Neuron::Neuron(Vector<float> &weights) :
    m_weights(weights) {}

Neuron::Neuron(Vector<float> &weights, std::function<float(float)> &lambda) :
    m_activation_fun(lambda), m_weights(weights) {}

void Neuron::adapt_gradient(Vector<float> &previous_layer_output, const float &dCdZ, Vector<float> &dCDZprime) {
    for (unsigned int j=0; j<previous_layer_output.size(); j++) {
        const float weight = m_weights[j];
        const float error = dCdZ * m_deriv_activation_fun(m_output);

        m_new_weights_delta[j] += error * previous_layer_output[j];
        m_new_bias_delta += dCdZ;

        dCDZprime[j] += error * weight;
    }
}

void Neuron::apply_gradient_delta(const float &epsilon, const float &max_gradiant) {
    float length = m_new_weights_delta.length();
    if (length > max_gradiant) {
        m_new_weights_delta /= length/max_gradiant;
    }
    m_weights += m_new_weights_delta * epsilon;
    m_new_weights_delta.assign(0);
    m_bias += m_new_bias_delta * epsilon;
    m_new_bias_delta = 0;
}

float Neuron::compute_output(Vector<float> input_vector) {
    float s = m_weights.dot(input_vector);
    m_output = m_activation_fun(s + m_bias);
    return m_output;
}

float Neuron::get_output() const {
    return m_output;
}
