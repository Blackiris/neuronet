#include "maxpool_layer.h"

MaxpoolLayer::MaxpoolLayer(const unsigned int &input_width, const unsigned int &input_height, const unsigned int &size)
    : INeuronsLayer(input_width*input_height/(size*size)), m_input_width(input_width), m_input_height(input_height), m_size(size) {}


Vector<float> MaxpoolLayer::compute_outputs(const Vector<float> &input_vector) {
    unsigned int output_width = m_input_width / m_size;
    unsigned int output_height = m_input_height / m_size;

    for (unsigned int output_x=0; output_x<output_width; output_x++) {
        for (unsigned int output_y=0; output_y<output_height; output_y++) {
            float max = input_vector[output_x*m_size+(output_y*m_size)*m_input_width];
            for (int i=0; i<(int)m_size; i++) {
                for (int j=0; j<(int)m_size; j++) {
                    const unsigned int input_x = output_x*m_size+i;
                    const unsigned int input_y = output_y*m_size+j;
                    if (input_x >= m_input_width || input_y >= m_input_height) {
                        continue;
                    }

                    float new_val = input_vector[input_x+input_y*m_input_width];
                    if (new_val > max) {
                        max = new_val;
                    }
                }
            }
            m_outputs[output_x+output_y*output_width] = max;
        }
    }

    return m_outputs;
}


void MaxpoolLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    const unsigned int output_x = m_input_width / m_size;

    for (unsigned int k=0; k<dCdZ.size(); k++) {
        unsigned int x_out = k%output_x;
        unsigned int y_out = k/output_x;

        int previous_idx_max = 0;
        float max = previous_layer_output[0];
        for (int i=0; i<(int)m_size; i++) {
            for (int j=0; j<(int)m_size; j++) {
                const unsigned int input_x = x_out*m_size+i;
                const unsigned int input_y = y_out*m_size+j;
                if (input_x >= m_input_width || input_y >= m_input_height) {
                    continue;
                }


                unsigned int previous_idx = input_x+input_y*m_input_width;
                if (previous_layer_output[previous_idx] > max) {
                    max = previous_layer_output[previous_idx];
                    previous_idx_max = previous_idx;
                }
            }
        }

        dCdZprime[previous_idx_max + dcdz_prime_offset] += dCdZ[k];
    }
}

void MaxpoolLayer::apply_new_weights(const TrainingParams &training_params) {
    // Nothing to do
}
