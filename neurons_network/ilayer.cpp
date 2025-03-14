#include "ilayer.h"

ILayer::ILayer() {}

ILayer::ILayer(const int &output_size) : m_outputs(output_size, 0.f) {

}

float ILayer::get_value_at(const int &pos) {
    return m_outputs[pos];
}
