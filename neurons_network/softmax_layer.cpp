#include "softmax_layer.h"

SoftmaxLayer::SoftmaxLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &size)
    : INeuronsLayer(input_x*input_y/(size*size)), m_input_x(input_x), m_input_y(input_y), m_size(size) {}


Vector<float> SoftmaxLayer::compute_outputs(const Vector<float> &input_vector) {
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


Vector<float> SoftmaxLayer::adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ, const float &epsilon) {
    unsigned int output_x = m_input_x / m_size;
    Vector<float> dCdZprime(previous_layer_output.size(), 0);

    for (unsigned int k=0; k<dCdZ.size(); k++) {
        unsigned int x_out = k%output_x;
        unsigned int y_out = k/output_x;

        float dCdZk = dCdZ[k];

        for (int i=0; i<(int)m_size; i++) {
            for (int j=0; j<(int)m_size; j++) {
                unsigned int previous_idx = x_out+i+(y_out+j)*m_input_x;
                dCdZprime[previous_idx] += dCdZk;
            }
        }
    }
    return dCdZprime;
}

void SoftmaxLayer::apply_new_weights(const float &max_gradiant) {
    // Nothing to do
}
