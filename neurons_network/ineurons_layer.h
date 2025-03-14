#ifndef INEURONS_LAYER_H
#define INEURONS_LAYER_H

#include "ilayer.h"
class INeuronsLayer : public ILayer
{
public:
    INeuronsLayer(const int &output_size);

    virtual void adapt_gradient(ILayer &previous_layer, Vector<float> &dCdZ, const float &epsilon, Vector<float> &dCdZprime) = 0;
    virtual void apply_new_weights(const float &max_gradiant) = 0;

};

#endif // INEURONS_LAYER_H
