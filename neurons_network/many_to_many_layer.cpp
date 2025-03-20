#include "many_to_many_layer.h"

ManyToManyLayer::ManyToManyLayer(std::vector<INeuronsLayer*> &sub_layers, const unsigned int &sub_input_size)
    : INeuronsLayer(sub_layers[0]->get_output_size()*sub_layers.size()), m_sub_input_size(sub_input_size), m_sub_output_size(sub_layers[0]->get_output_size()), m_sub_layers(sub_layers) {}


Vector<float> ManyToManyLayer::compute_outputs(const Vector<float> &input_vector) {
    unsigned int offset_input = 0;
    unsigned int offset_output = 0;
    for (auto& sub_layer: m_sub_layers) {

        Vector<float> sub_input_vector = Vector<float>(input_vector.begin() + offset_input,
                                                    input_vector.begin() + offset_input + m_sub_input_size);
        Vector<float> sub_res = sub_layer->compute_outputs(input_vector);

        m_outputs.copy(sub_res, offset_output);
        offset_input += m_sub_input_size;
        offset_output += sub_res.size();
    }
    return m_outputs;
}

Vector<float> ManyToManyLayer::adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ) {
    unsigned int offset_input = 0;
    unsigned int offset_output = 0;
    Vector<float> dCdZprime;
    for (auto& sub_layer: m_sub_layers) {
        Vector<float> sub_dCdZ(&dCdZ[offset_output], &dCdZ[offset_output+m_sub_output_size]);
        Vector<float> sub_previous_layer_output(&previous_layer_output[offset_input], &previous_layer_output[offset_input+m_sub_input_size]);
        Vector<float> sub_dCdZprime = sub_layer->adapt_gradient(sub_previous_layer_output, sub_dCdZ);

        offset_input += m_sub_input_size;
        offset_output += m_sub_output_size;
        dCdZprime.insert_back(sub_dCdZprime);
        //std::cout << sub_dCdZprime << "\n\n" << dCdZprime << "\n\n\n";
    }
    return dCdZprime;
}

void ManyToManyLayer::apply_new_weights(const float &epsilon, const float &max_gradiant) {
    for (auto& sub_layer: m_sub_layers) {
        sub_layer->apply_new_weights(epsilon, max_gradiant);
    }
}

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}
