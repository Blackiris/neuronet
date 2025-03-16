#ifndef ILAYER_H
#define ILAYER_H

#include "../vector.h"

class ILayer
{
public:
    ILayer(const int &output_size);
    virtual Vector<float> compute_outputs(const Vector<float> &input_vector) = 0;
    Vector<float> get_output();
    unsigned int get_output_size();
    float get_value_at(const int &pos);


protected:
    Vector<float> m_outputs;
};

#endif // ILAYER_H
