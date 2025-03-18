#ifndef NEURONS_NETWORK_H
#define NEURONS_NETWORK_H

#include <memory>
#include <vector>
#include "input_layer.h"
#include "ineurons_layer.h"

class NeuronsNetwork
{
public:
    NeuronsNetwork();
    Vector<float> compute(const Vector<float> &input);

    InputLayer m_input_layer;
    std::vector<std::unique_ptr<INeuronsLayer>> m_layers;

    void apply_new_weights(const float &epsilon, const float &max_gradiant);
};

#endif // NEURONS_NETWORK_H
