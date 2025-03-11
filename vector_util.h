#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include "vector.h"
namespace VectorUtil {
    template<typename T> T find_max(const Vector<T> &vector) {
        T max = 0;
        for (unsigned int i=0; i<vector.size(); i++) {
            const T &val = vector[i];
            if (val > max) {
                max = val;
            }
        }
        return max;
    }

    template<typename T> unsigned int find_max_pos(const Vector<T> &vector) {
        if (vector.size() == 0) {
            return -1;
        }
        unsigned int pos = 0;
        T max = 0;
        for (unsigned int i=0; i<vector.size(); i++) {
            const T &val = vector[i];
            if (val > max) {
                max = val;
                pos = i;
            }
        }
        return pos;
    }

}

#endif // VECTOR_UTIL_H
