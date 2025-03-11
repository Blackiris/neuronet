#ifndef ILAYER_H
#define ILAYER_H

#include "vector.h"

class ILayer
{
public:
    ILayer();
    ILayer(const int &output_size);
    virtual Vector<float> compute_outputs(const Vector<float> &input_vector) = 0;
    virtual unsigned int get_output_size() = 0;
    float get_value_at(const int &pos);

    void apply_new_weights();

protected:
    Vector<float> m_outputs;
};

#endif // ILAYER_H
