#ifndef ILAYER_H
#define ILAYER_H

#include "../vector.h"

class ILayer
{
public:
    ILayer(const int &output_size);
    virtual Vector<float> compute_outputs(const Vector<float> &input_vector) = 0;
    [[nodiscard]] const Vector<float>& get_output() const;
    [[nodiscard]] unsigned int get_output_size() const;
    [[nodiscard]] float get_value_at(const int &pos) const;


protected:
    Vector<float> m_outputs;
};

#endif // ILAYER_H
