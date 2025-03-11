#ifndef NEURON_H
#define NEURON_H

#include <functional>
#include "ioutput.h"
#include "vector.h"


class Neuron : public IOutput
{
public:
    Neuron(int nb_weight);
    Neuron(Vector<float> &weights);
    Neuron(Vector<float> &weights, std::function<float(float)> &lambda);

    float compute_output(Vector<float> input_vector);
    float get_output() const override;

    Vector<float> m_weights;
    Vector<float> m_new_weights_delta;


    void apply_weight_delta(const float &max_gradiant);
    std::function<float(float)> m_activation_fun = [](float x) { return x < 0 ? 0 : x; };
    std::function<float(float)> m_deriv_activation_fun = [](float x) { return x < 0 ? 0 : 1; };


private:
    float m_output = 0;
};

#endif // NEURON_H
