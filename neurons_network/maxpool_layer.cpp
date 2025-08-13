#include "maxpool_layer.h"

MaxpoolLayer::MaxpoolLayer(const unsigned int &input_width, const unsigned int &input_height, const unsigned int &nb_features_map, const unsigned int &size)
    : INeuronsLayer(nb_features_map * input_width*input_height/(size*size)), m_input_width(input_width), m_input_height(input_height),
    m_nb_features_map(nb_features_map), m_size(size), m_input_map_size(m_input_width*m_input_height), m_output_map_size(m_input_map_size/(m_size*m_size)) {

}


Vector<float> MaxpoolLayer::compute_outputs(const Vector<float> &input_vector) {
    const unsigned int output_width = m_input_width / m_size;
    const unsigned int output_height = m_input_height / m_size;
    unsigned int input_map_offset = 0;
    unsigned int output_map_offset = 0;

    for (unsigned int input_map_idx=0; input_map_idx<m_nb_features_map; input_map_idx++) {
        for (unsigned int output_x=0; output_x<output_width; output_x++) {
            for (unsigned int output_y=0; output_y<output_height; output_y++) {
                float max = input_vector[input_map_offset + output_x*m_size+(output_y*m_size)*m_input_width];
                for (int i=0; i<(int)m_size; i++) {
                    for (int j=0; j<(int)m_size; j++) {
                        const unsigned int input_x = output_x*m_size+i;
                        const unsigned int input_y = output_y*m_size+j;
                        if (input_x >= m_input_width || input_y >= m_input_height) {
                            continue;
                        }

                        float new_val = input_vector[input_map_offset + input_x+input_y*m_input_width];
                        if (new_val > max) {
                            max = new_val;
                        }
                    }
                }
                m_outputs[output_map_offset + output_x+output_y*output_width] = max;
            }
        }
        input_map_offset += m_input_map_size;
        output_map_offset += m_output_map_size;
    }

    return m_outputs;
}


void MaxpoolLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    const unsigned int output_width = m_input_width / m_size;

    unsigned int input_map_offset = 0;
    unsigned int output_map_offset = 0;

    for (unsigned int input_map_idx=0; input_map_idx<m_nb_features_map; input_map_idx++) {
        for (unsigned int output_idx=output_map_offset; output_idx<output_map_offset+m_output_map_size; output_idx++) {

            unsigned int output_local_idx = output_idx%m_output_map_size;
            unsigned int x_out = output_local_idx%output_width;
            unsigned int y_out = output_local_idx/output_width;

            int previous_idx_max = 0;
            float max = previous_layer_output[input_map_offset];
            for (int i=0; i<(int)m_size; i++) {
                for (int j=0; j<(int)m_size; j++) {
                    const unsigned int input_x = x_out*m_size+i;
                    const unsigned int input_y = y_out*m_size+j;
                    if (input_x >= m_input_width || input_y >= m_input_height) {
                        continue;
                    }


                    unsigned int previous_idx = input_map_offset+ input_x+input_y*m_input_width;
                    if (previous_layer_output[previous_idx] > max) {
                        max = previous_layer_output[previous_idx];
                        previous_idx_max = previous_idx;
                    }
                }
            }

            dCdZprime[previous_idx_max + dcdz_prime_offset] += dCdZ[output_idx];
        }

        input_map_offset += m_input_map_size;
        output_map_offset += m_output_map_size;
    }

}

void MaxpoolLayer::apply_new_weights(const TrainingParams &training_params) {
    // Nothing to do
}
