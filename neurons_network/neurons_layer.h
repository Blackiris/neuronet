#ifndef NEURONS_LAYER_H
#define NEURONS_LAYER_H

#include "ineurons_layer.h"
#include "neuron.h"
#include <vector>

class NeuronsLayer : public INeuronsLayer
{
public:
    NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights);

    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    Vector<float> adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ) override;
    void apply_new_weights(const float &epsilon, const float &max_gradiant) override;

    std::vector<Neuron> m_neurons;
};

#endif // NEURONS_LAYER_H
