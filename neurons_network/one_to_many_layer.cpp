#include "one_to_many_layer.h"

OneToManyLayer::OneToManyLayer(std::vector<std::unique_ptr<INeuronsLayer>> &&sub_layers)
    : INeuronsLayer(sub_layers[0]->get_output_size()*sub_layers.size()),
    m_sub_output_size(sub_layers[0]->get_output_size()), m_sub_layers(std::move(sub_layers)) {

}

Vector<float> OneToManyLayer::compute_outputs(const Vector<float> &input_vector) {
    unsigned int offset_output = 0;
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_res = sub_layer->compute_outputs(input_vector);
        m_outputs.copy(sub_res, offset_output);
        offset_output += sub_res.size();
    }
    return m_outputs;
}

void OneToManyLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) {
    unsigned int offset = 0;
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_dCdZ(dCdZ.begin() + offset, dCdZ.begin() + offset+m_sub_output_size);
        sub_layer->adapt_gradient(previous_layer_output, sub_dCdZ, dCdZprime, dcdz_prime_offset);
        offset += m_sub_output_size;
    }
    //std::cout << dCdZ << "\n\n" << dCdZprime << "\n";
}

void OneToManyLayer::apply_new_weights(const TrainingParams &training_params) {
    for (auto& sub_layer: m_sub_layers) {
        sub_layer->apply_new_weights(training_params);
    }
}
