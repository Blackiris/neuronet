#ifndef MANY_TO_MANY_LAYER_H
#define MANY_TO_MANY_LAYER_H

#include "ineurons_layer.h"

class ManyToManyLayer : public INeuronsLayer
{
public:
    ManyToManyLayer(std::vector<INeuronsLayer*> &sub_layers, const unsigned int &sub_input_size);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) override;
    void apply_new_weights(const float &epsilon, const float &max_gradiant) override;

private:
    unsigned int m_sub_input_size, m_sub_output_size;
    std::vector<INeuronsLayer*> m_sub_layers;
};

#endif // MANY_TO_MANY_LAYER_H
