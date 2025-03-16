#ifndef NEURON_H
#define NEURON_H

#include "../ioutput.h"
#include "../vector.h"
#include "ilayer.h"
#include <functional>
#include <random>

class Neuron : public IOutput
{
public:
    Neuron();
    Neuron(int nb_weight);
    Neuron(Vector<float> &weights);
    Neuron(Vector<float> &weights, std::function<float(float)> &lambda);

    float compute_output(Vector<float> input_vector);
    float get_output() const override;

    void adapt_gradient(Vector<float> &previous_layer_output, const float &dCdZ, const float &epsilon, Vector<float> &dCDZprime);
    void apply_gradient_delta(const float &max_gradiant);
    std::function<float(float)> m_activation_fun = [](float x) { return x < 0 ? 0 : x; };
    std::function<float(float)> m_deriv_activation_fun = [](float x) { return x < 0 ? 0 : 1; };

protected:
    float m_output = 0;

private:
    float m_bias = 0;
    float m_new_bias_delta;
    Vector<float> m_weights;
    Vector<float> m_new_weights_delta;

    static std::random_device rd;
    static std::mt19937 gen;
};

#endif // NEURON_H
