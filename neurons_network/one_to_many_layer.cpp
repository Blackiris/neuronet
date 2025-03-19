#include "one_to_many_layer.h"

OneToManyLayer::OneToManyLayer(std::vector<INeuronsLayer*> &sub_layers)
    : INeuronsLayer(sub_layers[0]->get_output_size()*sub_layers.size()), m_sub_output_size(sub_layers[0]->get_output_size()), m_sub_layers(sub_layers) {

}

Vector<float> OneToManyLayer::compute_outputs(const Vector<float> &input_vector) {
    Vector<float> result(m_sub_output_size*m_sub_layers.size());
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_res = sub_layer->compute_outputs(input_vector);
        result.insert(sub_res);
    }
    return result;
}

Vector<float> OneToManyLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ) {
    unsigned int offset = 0;
    Vector<float> dCdZprime(previous_layer_output.size(), 0);
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_dCdZ(dCdZ.begin() + offset, dCdZ.begin() + offset+m_sub_output_size);
        Vector<float> sub_dCdZprime = sub_layer->adapt_gradient(previous_layer_output, sub_dCdZ);
        dCdZprime += sub_dCdZprime;
        offset += m_sub_output_size;
    }
    //std::cout << dCdZ << "\n\n" << dCdZprime << "\n";
    return dCdZprime;
}

void OneToManyLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (auto& sub_layer: m_sub_layers) {
        sub_layer->apply_new_weights(epsilon, max_gradiant);
    }
}
