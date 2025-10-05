#ifndef ILAYER_H
#define ILAYER_H

#include "vector.h"

class ILayer
{
public:
    explicit ILayer(const int &output_size);
    virtual Vector<float> compute_outputs(const Vector<float> &input_vector) = 0;

    [[nodiscard]] inline float get_value_at(const int &pos) const {
        return m_outputs[pos];
    }

    [[nodiscard]] inline const Vector<float>& get_output() const {
        return m_outputs;
    }

    [[nodiscard]] inline unsigned int get_output_size() const {
        return m_outputs.size();
    }


protected:
    Vector<float> m_outputs;
};

#endif // ILAYER_H
