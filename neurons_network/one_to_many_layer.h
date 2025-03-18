#ifndef ONE_TO_MANY_LAYER_H
#define ONE_TO_MANY_LAYER_H

#include "ineurons_layer.h"

class OneToManyLayer : public INeuronsLayer
{
public:
    OneToManyLayer(std::vector<INeuronsLayer*> &sub_layers);
    Vector<float> compute_outputs(const Vector<float> &input_vector) override;

    Vector<float> adapt_gradient(Vector<float> &previous_layer_output, Vector<float> &dCdZ) override;
    void apply_new_weights(const float &epsilon, const float &max_gradiant) override;

private:
    unsigned int m_sub_output_size;
    std::vector<INeuronsLayer*> m_sub_layers;
};

#endif // ONE_TO_MANY_LAYER_H
