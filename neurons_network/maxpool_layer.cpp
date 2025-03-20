#include "maxpool_layer.h"

MaxpoolLayer::MaxpoolLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &size)
    : INeuronsLayer(input_x*input_y/(size*size)), m_input_x(input_x), m_input_y(input_y), m_size(size) {}


Vector<float> MaxpoolLayer::compute_outputs(const Vector<float> &input_vector) {
    unsigned int output_x = m_input_x / m_size;
    unsigned int output_y = m_input_y / m_size;

    for (unsigned int x=0; x<output_x; x++) {
        for (unsigned int y=0; y<output_y; y++) {
            float max = input_vector[x*m_size+(y*m_size)*m_input_x];
            for (int i=0; i<(int)m_size; i++) {
                for (int j=0; j<(int)m_size; j++) {
                    float new_val = input_vector[x+i*m_size+((y+j)*m_size)*m_input_x];
                    if (new_val > max) {
                        max = new_val;
                    }
                }
            }
            m_outputs[x+y*output_x] = max;
        }
    }

    return m_outputs;
}


void MaxpoolLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    const unsigned int output_x = m_input_x / m_size;

    for (unsigned int k=0; k<dCdZ.size(); k++) {
        unsigned int x_out = k%output_x;
        unsigned int y_out = k/output_x;

        int previous_idx_max = 0;
        float max = previous_layer_output[0];
        for (int i=0; i<(int)m_size; i++) {
            for (int j=0; j<(int)m_size; j++) {
                unsigned int previous_idx = x_out+i+(y_out+j)*m_input_x;
                if (previous_layer_output[previous_idx] > max) {
                    max = previous_layer_output[previous_idx];
                    previous_idx_max = previous_idx;
                }
            }
        }

        dCdZprime[previous_idx_max + dcdz_prime_offset] += dCdZ[k];
    }
}

void MaxpoolLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    // Nothing to do
}
