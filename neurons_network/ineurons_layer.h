#ifndef INEURONS_LAYER_H
#define INEURONS_LAYER_H

#include "ilayer.h"
#include "training_data.h"

class INeuronsLayer : public ILayer
{
public:
    explicit INeuronsLayer(const int &output_size);
    virtual ~INeuronsLayer() noexcept = default;

    virtual void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &current_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) = 0;
    virtual void apply_new_weights(const TrainingParams &training_params) = 0;

};

#endif // INEURONS_LAYER_H
