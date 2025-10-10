#ifndef ILAYER_H
#define ILAYER_H

#include "vector.h"

class ILayer
{
public:
    explicit ILayer(const size_t &output_size);
    virtual ~ILayer() noexcept = default;
    virtual Vector<float> compute_outputs(const Vector<float> &input_vector) = 0;

    [[nodiscard]] inline unsigned int get_output_size() const {
        return m_output_size;
    }


protected:
    size_t m_output_size;
};

#endif // ILAYER_H
