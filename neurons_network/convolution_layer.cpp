#include "convolution_layer.h"
#include <random>

ConvolutionLayer::ConvolutionLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &conv_radius)
    : INeuronsLayer((input_x-2*conv_radius)*(input_y-2*conv_radius)), m_conv_weights(2*conv_radius+1, std::vector<float>(2*conv_radius+1, 0)),
    m_conv_weights_delta(2*conv_radius+1, std::vector<float>(2*conv_radius+1, 0)),
    m_conv_radius(conv_radius), m_input_x(input_x), m_input_y(input_y) {

    int conv_side_length = 2*conv_radius+1;

    // Xavier - He init
    std::normal_distribution d{0.0, 2.0/(conv_side_length*conv_side_length)};
    std::mt19937 gen;

    for (int i=0; i<conv_side_length; i++) {
        for (int j=0; j<conv_side_length; j++) {
            float r = d(gen);
            m_conv_weights[i][j] = r;
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
            m_outputs[x-m_conv_radius+(y-m_conv_radius)*output_x] = val;
        }
    }

    return m_outputs;
}


Vector<float> ConvolutionLayer::adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ) {
    unsigned int output_x = m_input_x - 2*m_conv_radius;
    Vector<float> dCdZprime(previous_layer_output.size(), 0);

    for (unsigned int k=0; k<dCdZ.size(); k++) {
        unsigned int x_out = k%output_x;
        unsigned int y_out = k/output_x;

        float dCdZk = dCdZ[k];

        for (int i=-m_conv_radius; i<=(int)m_conv_radius; i++) {
            for (int j=-m_conv_radius; j<=(int)m_conv_radius; j++) {
                const float weight = m_conv_weights[i+m_conv_radius][j+m_conv_radius];

                unsigned int previous_idx = x_out+i+m_conv_radius+(y_out+j+m_conv_radius)*m_input_x;
                m_conv_weights_delta[i+m_conv_radius][j+m_conv_radius] += dCdZk * previous_layer_output[previous_idx];
                //std::cout << dCdZk << "-"<< previous_layer_output[previous_idx] << "\n";
                dCdZprime[previous_idx] += dCdZk * weight;
            }
        }
    }

    return dCdZprime;
}

void ConvolutionLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (int i=0; i<=(int)m_conv_radius*2; i++) {
        for (int j=0; j<=(int)m_conv_radius*2; j++) {
            //std::cout << m_conv_weights_delta[i][j] << "\n" <<std::endl;
            m_conv_weights[i][j] += epsilon*m_conv_weights_delta[i][j];
            m_conv_weights_delta[i][j] = 0;
        }
    }
}
