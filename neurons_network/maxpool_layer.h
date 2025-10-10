#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "ineurons_layer.h"

class MaxpoolLayer : public INeuronsLayer
{
public:
    MaxpoolLayer(const unsigned int &input_width, const unsigned int &input_height, const unsigned int &nb_features_map, const unsigned int &size);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;
    void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &current_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) override;
    void apply_new_weights(const TrainingParams &training_params) override;

private:
    const unsigned int m_input_width, m_input_height, m_nb_features_map;
    const unsigned int m_size, m_input_map_size, m_output_map_size;
};

#endif // MAXPOOL_LAYER_H
