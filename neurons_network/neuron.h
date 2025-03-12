#ifndef NEURON_H
#define NEURON_H

#include "../ioutput.h"
#include "../vector.h"
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

    float& get_weight(const unsigned int &idx);

    void add_weight_delta(const unsigned int &idx, const float &delta_to_add);
    void apply_weight_delta(const float &max_gradiant);
    std::function<float(float)> m_activation_fun = [](float x) { return x < 0 ? 0 : x; };
    std::function<float(float)> m_deriv_activation_fun = [](float x) { return x < 0 ? 0 : 1; };

protected:
    float m_output = 0;

private:
    Vector<float> m_weights;
    Vector<float> m_new_weights_delta;

    static std::random_device rd;
    static std::mt19937 gen;
};

#endif // NEURON_H
