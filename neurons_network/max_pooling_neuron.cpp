#include "max_pooling_neuron.h"

MaxPoolingNeuron::MaxPoolingNeuron(const unsigned int &offset, const unsigned int &input_x, const unsigned int &input_y,
                                   const unsigned int &pool_x, const unsigned int &pool_y, const unsigned int &pool_size)
    : m_offset(offset), m_input_x(input_x), m_input_y(input_y), m_pool_x(pool_x), m_pool_y(pool_y), m_pool_size(pool_size) {}


float MaxPoolingNeuron::compute_output(Vector<float> input_vector) {
    m_output = input_vector[m_offset+m_pool_y*m_input_x+m_pool_x];

    for (unsigned int i = m_pool_x; i<m_pool_x+m_pool_size; i++) {
        for (unsigned int j = m_pool_y; j<m_pool_y+m_pool_size; j++) {
            float val = input_vector[m_offset+j*m_input_x+i];
            if (val > m_output) {
                m_output = val;
            }
        }
    }

    return m_output;
}

void MaxPoolingNeuron::adapt_gradient(ILayer &previous_layer, const float &dCdZ, const float &epsilon, Vector<float> &dCDZprime) {
    for (unsigned int i = m_pool_x; i<m_pool_x+m_pool_size; i++) {
        for (unsigned int j = m_pool_y; j<m_pool_y+m_pool_size; j++) {
            dCDZprime[m_offset+j*m_input_x+i] += dCdZ;
        }
    }
}
