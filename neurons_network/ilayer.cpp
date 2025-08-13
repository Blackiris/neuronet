#include "ilayer.h"


ILayer::ILayer(const int &output_size) : m_outputs(output_size, 0.f) {}

float ILayer::get_value_at(const int &pos) const {
    return m_outputs[pos];
}

const Vector<float>& ILayer::get_output() const {
    return m_outputs;
}

unsigned int ILayer::get_output_size() const {
    return m_outputs.size();
}
