#include "convolution_layer.h"
#include <random>

std::random_device ConvolutionLayer::rd;
std::mt19937 ConvolutionLayer::gen(ConvolutionLayer::rd());

ConvolutionLayer::ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_radius)
    : INeuronsLayer((input_x-2*conv_radius)*(input_y-2*conv_radius)), m_conv_weights(2*conv_radius+1, std::vector<float>(2*conv_radius+1, 0)),
    m_conv_weights_delta(2*conv_radius+1, std::vector<float>(2*conv_radius+1, 0)),
    m_conv_radius(conv_radius), m_input_x(input_x), m_input_y(input_y) {

    const unsigned int conv_side = 2*m_conv_radius + 1;
    const unsigned int conv_side_squared = conv_side * conv_side;

    // Xavier - He init
    std::normal_distribution d{0.0, std::sqrt(2.0/(m_input_x*m_input_y*conv_side_squared))};

    float mean = 0;
    for (unsigned int i=0; i<conv_side; i++) {
        for (unsigned int j=0; j<conv_side; j++) {
            float r = d(gen);
            m_conv_weights[i][j] = r;
            mean += r;
        }
    }

    mean /= conv_side_squared;
    for (unsigned int i=0; i<conv_side; i++) {
        for (unsigned int j=0; j<conv_side; j++) {
            m_conv_weights[i][j] -= mean;
        }
    }
}

Vector<float> ConvolutionLayer::compute_outputs(const Vector<float> &input_vector) {
    unsigned int output_x = m_input_x - 2*m_conv_radius;
    for (unsigned int x=m_conv_radius; x<m_input_x-m_conv_radius; x++) {
        for (unsigned int y=m_conv_radius; y<m_input_y-m_conv_radius; y++) {
            float val{0};
            for (int i=-m_conv_radius; i<=(int)m_conv_radius; i++) {
                for (int j=-m_conv_radius; j<=(int)m_conv_radius; j++) {
                    val += input_vector[x+i+(y+j)*m_input_x]*m_conv_weights[i+m_conv_radius][j+m_conv_radius];
                }
            }
            m_outputs[x-m_conv_radius+(y-m_conv_radius)*output_x] = m_activation_fun(val);
        }
    }

    return m_outputs;
}


void ConvolutionLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    const unsigned int output_x = m_input_x - 2*m_conv_radius;

    for (unsigned int k=0; k<dCdZ.size(); k++) {
        const unsigned int x_out = k%output_x;
        const unsigned int y_out = k/output_x;

        const float error = dCdZ[k] * m_deriv_activation_fun(m_outputs[k]);

        for (int i=-m_conv_radius; i<=(int)m_conv_radius; i++) {
            for (int j=-m_conv_radius; j<=(int)m_conv_radius; j++) {
                const unsigned int weight_i = i+m_conv_radius;
                const unsigned int weight_j = j+m_conv_radius;
                const float weight = m_conv_weights[weight_i][weight_j];

                unsigned int previous_idx = x_out+weight_i+(y_out+weight_j)*m_input_x;
                m_conv_weights_delta[weight_i][weight_j] += error * previous_layer_output[previous_idx];
                //std::cout << k << " - error:" << error << " - prevval:"<< previous_layer_output[previous_idx] << "\n";
                dCdZprime[previous_idx + dcdz_prime_offset] += error * weight;
            }
        }
    }
}

void ConvolutionLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (int i=0; i<=(int)m_conv_radius*2; i++) {
        for (int j=0; j<=(int)m_conv_radius*2; j++) {
            m_conv_weights[i][j] += epsilon*m_conv_weights_delta[i][j];
            m_conv_weights_delta[i][j] = 0;
        }
    }
}
