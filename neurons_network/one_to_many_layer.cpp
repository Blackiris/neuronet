#include "one_to_many_layer.h"

OneToManyLayer::OneToManyLayer(std::vector<INeuronsLayer> &sub_layers)
    : INeuronsLayer(sub_layers[0].get_output_size()*sub_layers.size()), m_sub_output_size(sub_layers[0].get_output_size()), m_sub_layers(sub_layers) {

}

Vector<float> OneToManyLayer::compute_outputs(const Vector<float> &input_vector) {
    Vector<float> result(m_sub_output_size*m_sub_layers.size());
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_res = sub_layer.compute_outputs(input_vector);
        result.insert(sub_res);
    }
    return result;
}
