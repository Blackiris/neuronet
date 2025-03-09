#include "vector_util.h"

template<typename T> T find_max(const Vector<T> &vector) {
    T max = 0;
    for (auto& val: vector) {
        if (val > max) {
            max = val;
        }
    }
    return max;
}
