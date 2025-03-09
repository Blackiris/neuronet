#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "ilayer.h"

class InputLayer : public ILayer
{
public:
    InputLayer();
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;
    unsigned int get_output_size() override;

};

#endif // INPUT_LAYER_H
