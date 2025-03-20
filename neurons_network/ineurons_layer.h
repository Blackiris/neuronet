#ifndef INEURONS_LAYER_H
#define INEURONS_LAYER_H

#include "ilayer.h"
class INeuronsLayer : public ILayer
{
public:
    INeuronsLayer(const int &output_size);

    virtual void adapt_gradient(const Vector<float> &previous_layer_output, const Vector<float> &dCdZ, Vector<float> &dCdZprime, const unsigned int &dcdz_prime_offset) = 0;
    virtual void apply_new_weights(const float &epsilon, const float &max_gradiant) = 0;

};

#endif // INEURONS_LAYER_H
