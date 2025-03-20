#include "neurons_layer.h"

std::random_device NeuronsLayer::rd;
std::mt19937 NeuronsLayer::gen(NeuronsLayer::rd());


NeuronsLayer::NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights): INeuronsLayer(size), m_biases(size, 0.f), m_biases_delta(size, 0.f) {
    // Xavier - He init
    std::normal_distribution d{0.0, std::sqrt(2.0/nb_weights)};
    for (unsigned int i=0; i<size; i++) {
        Vector<float> m_weights(nb_weights, 0.f);
        float avg = 0;
        for (unsigned int j=0; j<nb_weights; j++) {
            float r = d(gen);
            m_weights[j] = r;
            avg += r;
        }
        avg /= nb_weights;

        for (unsigned int j=0; j<nb_weights; j++) {
            m_weights[j] -= avg;
        }

        m_weights_mat.emplace_back(m_weights);
        m_weights_mat_delta.emplace_back(Vector<float>(nb_weights, 0.f));
    }
}

Vector<float> NeuronsLayer::compute_outputs(const Vector<float> &input_vector) {
    for (unsigned int i=0; i<m_weights_mat.size(); i++) {
        float s = m_weights_mat[i].dot(input_vector);
        m_outputs[i] = m_activation_fun(s + m_biases[i]);
    }
    return m_outputs;
}

void NeuronsLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    for (unsigned int i=0; i<m_weights_mat.size(); i++) {
        const float error = dCdZ[i] * m_deriv_activation_fun(m_outputs[i]);
        for (unsigned int j=0; j<previous_layer_output.size(); j++) {
            const float weight = m_weights_mat[i][j];

            m_weights_mat_delta[i][j] += error * previous_layer_output[j];

            dCdZprime[j+dcdz_prime_offset] += error * weight;
        }
        m_biases_delta[i] += dCdZ[i];

        //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
    }
}

void NeuronsLayer::apply_new_weights(const float &epsilon, const float &clip_gradiant_threshold) {
    for (unsigned int i=0; i<m_weights_mat.size(); i++) {
        float length = m_weights_mat_delta[i].length();
        m_weights_mat_delta[i] *= epsilon;
        if (clip_gradiant_threshold> 0 && length > clip_gradiant_threshold) {
            m_weights_mat_delta[i] /= length/clip_gradiant_threshold;
        }
        //std::cout << m_weights.length() << "  -  " << m_new_weights_delta.length() << "\n";
        m_weights_mat[i] += m_weights_mat_delta[i];
        m_weights_mat_delta[i].assign(0);
        m_biases[i] += m_biases_delta[i] * epsilon *0.01;
        m_biases_delta[i] = 0;
    }
}
