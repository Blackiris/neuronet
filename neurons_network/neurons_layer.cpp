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
    for (unsigned int output_idx=0; output_idx<m_weights_mat.size(); output_idx++) {
        const float error = dCdZ[output_idx] * m_deriv_activation_fun(m_outputs[output_idx]);
        for (unsigned int prev_idx=0; prev_idx<previous_layer_output.size(); prev_idx++) {
            const float weight = m_weights_mat[output_idx][prev_idx];

            m_weights_mat_delta[output_idx][prev_idx] += error * previous_layer_output[prev_idx];

            dCdZprime[prev_idx+dcdz_prime_offset] += error * weight;
        }
        m_biases_delta[output_idx] += dCdZ[output_idx];

        //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
    }
}

void NeuronsLayer::apply_new_weights(const TrainingParams &training_params) {
    for (unsigned int i=0; i<m_weights_mat.size(); i++) {
        m_weights_mat_delta[i] *= training_params.epsilon;
        float length = m_weights_mat_delta[i].length();
        if (training_params.clip_gradiant_threshold> 0 && length > training_params.clip_gradiant_threshold) {
            m_weights_mat_delta[i] /= length/training_params.clip_gradiant_threshold;
        }
        //std::cout << m_weights.length() << "  -  " << m_new_weights_delta.length() << "\n";
        m_weights_mat[i] += m_weights_mat_delta[i];
        m_weights_mat_delta[i].assign(0);
        m_biases[i] += m_biases_delta[i] * training_params.epsilon_bias;
        m_biases_delta[i] = 0;
    }
}
