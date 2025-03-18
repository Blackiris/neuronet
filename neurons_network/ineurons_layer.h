#ifndef INEURONS_LAYER_H
#define INEURONS_LAYER_H

#include "ilayer.h"
class INeuronsLayer : public ILayer
{
public:
    INeuronsLayer(const int &output_size);

    virtual Vector<float> adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ) = 0;
    virtual void apply_new_weights(const float &epsilon, const float &max_gradiant) = 0;

};

#endif // INEURONS_LAYER_H
