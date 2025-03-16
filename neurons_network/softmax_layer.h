#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "ineurons_layer.h"

class SoftmaxLayer : public INeuronsLayer
{
public:
    SoftmaxLayer(const unsigned int &input_x, const unsigned int &input_y, const unsigned int &size);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;
    Vector<float> adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ, const float &epsilon) override;
    void apply_new_weights(const float &max_gradiant) override;

private:
    unsigned int m_input_x, m_input_y;
    const unsigned int m_size;
};

#endif // SOFTMAX_LAYER_H
