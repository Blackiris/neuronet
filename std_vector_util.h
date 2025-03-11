#ifndef STD_VECTOR_UTIL_H
#define STD_VECTOR_UTIL_H

#include <memory>
#include <algorithm>
#include <vector>

namespace StdVectorUtil {
    template<typename T> std::vector<std::vector<T>> split_chunks(const std::vector<T> &vector, const unsigned int& chunk_size) {
        std::vector<std::vector<T>> res;
        size_t vector_size = vector.size();
        for(size_t i = 0; i < vector_size; i += chunk_size) {
            auto last = std::min(i+chunk_size, vector_size);
            res.emplace_back(std::vector<T>(&vector[i], &vector[last]));
        }
        return res;
    }

}

#endif // STD_VECTOR_UTIL_H
