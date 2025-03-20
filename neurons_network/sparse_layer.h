#ifndef SPARSE_LAYER_H
#define SPARSE_LAYER_H

#include "ineurons_layer.h"

class SparseLayer : public INeuronsLayer
{
public:
    SparseLayer(std::vector<std::vector<bool>> link_table, std::vector<INeuronsLayer*> &sub_layers);

    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) override;
    void apply_new_weights(const float &epsilon, const float &max_gradiant) override;
};

#endif // SPARSE_LAYER_H
