#ifndef MAX_POOLING_NEURON_H
#define MAX_POOLING_NEURON_H

#include "neuron.h"

class MaxPoolingNeuron : public Neuron
{
public:
    MaxPoolingNeuron(const unsigned int &offset, const unsigned int &input_x, const unsigned int &input_y,
                     const unsigned int &pool_x, const unsigned int &pool_y, const unsigned int &pool_size);

    float compute_output(Vector<float> input_vector);
    void adapt_gradient(ILayer &previous_layer, const float &dCdZ, const float &epsilon, Vector<float> &dCDZprime);

private:
    unsigned int m_offset;
    unsigned int m_input_x, m_input_y;
    unsigned int m_pool_x, m_pool_y;
    unsigned int m_pool_size;
};

#endif // MAX_POOLING_NEURON_H
