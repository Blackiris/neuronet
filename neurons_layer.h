#ifndef NEURONS_LAYER_H
#define NEURONS_LAYER_H

#include <vector>
#include "neuron.h"
#include "ilayer.h"

class NeuronsLayer : public ILayer
{
public:
    NeuronsLayer(const unsigned int &size, const unsigned int &nb_weights);

    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    std::vector<Neuron> m_neurons;
};

#endif // NEURONS_LAYER_H
