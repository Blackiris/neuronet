#include "convolution_layer.h"
#include <random>

std::random_device ConvolutionLayer::rd;
std::mt19937 ConvolutionLayer::gen(ConvolutionLayer::rd());

ConvolutionLayer::ConvolutionLayer(const unsigned &input_x, const unsigned &input_y, const unsigned &conv_radius, std::vector<unsigned int> links_table)
    : INeuronsLayer((input_x-2*conv_radius)*(input_y-2*conv_radius)), m_links_table(links_table),
    m_conv_radius(conv_radius), m_conv_diameter(2*m_conv_radius+1), m_input_x(input_x), m_input_y(input_y), biases(m_links_table.size(), 0), biases_delta(biases.size(), 0)  {

    m_output_x = m_input_x - 2*m_conv_radius;
    const unsigned int conv_side = 2*m_conv_radius + 1;
    const unsigned int conv_side_squared = conv_side * conv_side;
    const int nb_links = m_links_table.size();

    const unsigned int m_conv_diameter_squared = m_conv_diameter * m_conv_diameter;
    variance_moment = Vector<float>(m_conv_diameter_squared, 0);

    // Xavier - He init
    std::normal_distribution d{0.0, std::sqrt(2.0/(m_input_x*m_input_y*conv_side_squared*nb_links))};

    for (unsigned int link_idx=0; link_idx<m_links_table.size(); link_idx++) {
        Vector<float> conv_weights_link(m_conv_diameter_squared, 0);
        float mean = 0;
        for (unsigned int i=0; i<m_conv_diameter_squared; i++) {
            float r = d(gen);
            conv_weights_link[i] = r;
            mean += r;
        }

        mean /= conv_side_squared;
        for (unsigned int i=0; i<m_conv_diameter_squared; i++) {
            conv_weights_link[i] -= mean;
        }

        m_conv_weights.emplace_back(conv_weights_link);
        m_conv_weights_delta.emplace_back(Vector<float>(m_conv_diameter_squared, 0));
        m_weights_momentum.emplace_back(Vector<float>(m_conv_diameter_squared, 0));
    }
}

Vector<float> ConvolutionLayer::compute_outputs(const Vector<float> &input_vector) {
    const unsigned int sub_input_size = m_input_x * m_input_y;

    for (unsigned int link_idx=0; link_idx<m_links_table.size(); link_idx++) {
        unsigned int link_offset = m_links_table[link_idx]*sub_input_size;

        for (unsigned int x=m_conv_radius; x<m_input_x-m_conv_radius; x++) {
            for (unsigned int y=m_conv_radius; y<m_input_y-m_conv_radius; y++) {
                float val{0};
                for (int i=-m_conv_radius; i<=(int)m_conv_radius; i++) {
                    for (int j=-m_conv_radius; j<=(int)m_conv_radius; j++) {
                        val += input_vector[link_offset + x+i+(y+j)*m_input_x]
                               * m_conv_weights[link_idx][i+m_conv_radius+(j+m_conv_radius)*m_conv_diameter];
                    }
                }
                const unsigned int output_idx = x-m_conv_radius+(y-m_conv_radius)*m_output_x;
                m_outputs[output_idx] = m_activation_fun(val + biases[link_idx]);
            }
        }
    }



    return m_outputs;
}


void ConvolutionLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    const unsigned int sub_input_size = m_input_x * m_input_y;

    for (unsigned int output_idx=0; output_idx<dCdZ.size(); output_idx++) {
        const unsigned int x_out = output_idx%m_output_x;
        const unsigned int y_out = output_idx/m_output_x;

        const float error = dCdZ[output_idx] * m_deriv_activation_fun(m_outputs[output_idx]);

        for (unsigned int link_idx=0; link_idx<m_links_table.size(); link_idx++) {
            unsigned int link_offset = m_links_table[link_idx]*sub_input_size;

            for (int i=-m_conv_radius; i<=(int)m_conv_radius; i++) {
                for (int j=-m_conv_radius; j<=(int)m_conv_radius; j++) {
                    const unsigned int weight_i = i+m_conv_radius;
                    const unsigned int weight_j = j+m_conv_radius;
                    const unsigned int weight_idx = weight_i+weight_j*m_conv_diameter;
                    const float weight = m_conv_weights[link_idx][weight_idx];

                    unsigned int previous_idx = link_offset + x_out+weight_i+(y_out+weight_j)*m_input_x;
                    m_conv_weights_delta[link_idx][weight_idx] += error * previous_layer_output[previous_idx];
                    //std::cout << k << " - error:" << error << " - prevval:"<< previous_layer_output[previous_idx] << "\n";
                    dCdZprime[previous_idx + dcdz_prime_offset] += error * weight;
                    biases_delta[link_idx] += dCdZ[output_idx];
                }
            }
        }

    }
}

void ConvolutionLayer::apply_new_weights(const TrainingParams &training_params) {
    for (unsigned int link_idx=0; link_idx<m_links_table.size(); link_idx++) {
        // Gradiant Descent
        //m_conv_weights[link_idx] += m_conv_weights_delta[link_idx] * training_params.epsilon;


        //adam
        m_weights_momentum[link_idx] = m_weights_momentum[link_idx] * training_params.adam_decay_rate_momentum
                                       + m_conv_weights_delta[link_idx] * (1-training_params.adam_decay_rate_momentum);
        variance_moment[link_idx] = variance_moment[link_idx] * training_params.adam_decay_rate_squared
                             + (1-training_params.adam_decay_rate_squared) * m_conv_weights_delta[link_idx].length_squared();
        const Vector<float> momentum_corrected = m_weights_momentum[link_idx]
                                                 /(1.f-std::pow(training_params.adam_decay_rate_momentum, training_params.current_epoch));
        const float variant_corrected = variance_moment[link_idx]
                                        /(1.f-std::pow(training_params.adam_decay_rate_squared, training_params.current_epoch));
        m_conv_weights[link_idx] += momentum_corrected/(std::sqrt(variant_corrected+1e-7)) * training_params.epsilon;


        m_conv_weights_delta[link_idx].assign(0);

        biases[link_idx] += biases_delta[link_idx] * training_params.epsilon_bias;
        biases_delta[link_idx] = 0;
    }
}
