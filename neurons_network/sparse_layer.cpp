#include "sparse_layer.h"



SparseLayer::SparseLayer(std::vector<std::vector<bool>> link_table, std::vector<INeuronsLayer*> &sub_layers)
    : INeuronsLayer(1) {}


Vector<float> SparseLayer::compute_outputs(const Vector<float> &input_vector) {
    return Vector<float>(0,0);
}

void SparseLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset)  {

}
void SparseLayer::apply_new_weights(const float &epsilon, const float &max_gradiant)  {

}
