#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "ilayer.h"

class InputLayer : public ILayer
{
public:
    InputLayer() noexcept;
    [[nodiscard]] Vector<float> compute_outputs(const Vector<float> &input_vector) override;

};

#endif // INPUT_LAYER_H
