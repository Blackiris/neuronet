#ifndef INEURONS_LAYER_H
#define INEURONS_LAYER_H

#include "ilayer.h"
class INeuronsLayer : public ILayer
{
public:
    INeuronsLayer();
    INeuronsLayer(const int &output_size);

    virtual void apply_new_weights(const float &max_gradiant) = 0;

};

#endif // INEURONS_LAYER_H
