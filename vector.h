#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
class Vector
{
public:
    constexpr Vector() noexcept;
    constexpr explicit Vector(const int &size);
    Vector(const int &size, T default_value);
    Vector(const std::vector<T> &vect);
    Vector(std::initializer_list<T> init);


    Vector operator+(const Vector& other) const {
        Vector res(this->m_vect);
        for (size_t i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] + other.m_vect[i];
        }
        return res;
    }

    Vector& operator+=(const Vector& other) {
        for (size_t i=0; i<m_vect.size(); i++) {
            this->m_vect[i] = this->m_vect[i] + other.m_vect[i];
        }
        return *this;
    }

    Vector operator-(const Vector& other) const {
        Vector res(this->m_vect);
        for (size_t i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] - other.m_vect[i];
        }
        return res;
    }

    Vector& operator-=(const Vector& other) {
        for (size_t i=0; i<m_vect.size(); i++) {
            this->m_vect[i] = this->m_vect[i] - other.m_vect[i];
        }
        return *this;
    }

    Vector operator*(const float& value) {
        Vector res(this->m_vect);
        for (size_t i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] * value;
        }
        return res;
    }

    Vector& operator*=(const float& value) {
        for (size_t i=0; i<m_vect.size(); i++) {
            this->m_vect[i] = this->m_vect[i] * value;
        }
        return *this;
    }

    Vector operator/(const float& value) const {
        Vector res(this->m_vect);
        for (size_t i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] / value;
        }
        return res;
    }

    Vector& operator/=(const float& value) {
        for (size_t i=0; i<m_vect.size(); i++) {
            this->m_vect[i] = this->m_vect[i] / value;
        }
        return *this;
    }



    T& operator[](const int& pos) {
        return this->m_vect[pos];
    }

    const T& operator[](const int& pos) const {
        return this->m_vect[pos];
    }

    size_t size() const {
        return m_vect.size();
    }

    void reserve(const int& size) {
        this->m_vect.reserve(size);
    }

    void assign(T* begin, T* end) {
        this->m_vect.assign(begin, end);
    }

    void assign(const T& value) {
        this->m_vect.assign(this->m_vect.size(), value);
    }

    T dot(const Vector<T>& other) {
        return std::inner_product(this->m_vect.cbegin(), this->m_vect.cend(), other.m_vect.cbegin(), 0);
    }

    double length() {
        return std::sqrt(length_squared());
    }

    double length_squared() {
        double sum = 0;
        for (auto& val : this->m_vect) {
            sum += val*val;
        }
        return sum;
    }

    void normalize() {
        *this = (*this/this->length());
    }

    void push(T value) {
        this->m_vect.emplace_back(value);
    }

    void insert_back(const Vector<T>& other) {
        this->m_vect.insert(this->m_vect.end(), other.m_vect.begin(), other.m_vect.end());
    }

    void copy(const Vector<T>& other, const unsigned int &from_index) {
        std::copy(other.begin(), other.end(), this->m_vect.begin() + from_index);
    }



    friend std::ostream& operator<<(std::ostream& os, const Vector<T>& dt) {
        os << '(';
        for(auto it = dt.m_vect.begin(); it != dt.m_vect.end() ; ++it) {
            os << *it;
            if (it < dt.m_vect.end() - 1) {
                os << ',';
            }
        }
        os << ')';
        return os;
    }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() {
        return m_vect.begin();
    }

    iterator end() {
        return m_vect.end();
    }

    const_iterator begin() const {
        return m_vect.begin();
    }

    const_iterator end() const {
        return m_vect.end();
    }



    Vector(T *begin, T *end);
    Vector(const iterator &begin, const iterator &end);

    template <typename InputIterator>
    Vector(InputIterator first, InputIterator last) : m_vect(first, last) {}


private:
    std::vector<T> m_vect;
};

template <typename T>
constexpr Vector<T>::Vector() noexcept {}

template <typename T>
constexpr Vector<T>::Vector(const int &size) {
    m_vect.reserve(size);
}

template <typename T>
Vector<T>::Vector(const int &size, T default_value) {
    m_vect.assign(size, default_value);
}

template <typename T>
Vector<T>::Vector(const std::vector<T> &vect): m_vect(vect) {}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> init) {
    m_vect.assign(init.begin(), init.end());
}

template <typename T>
Vector<T>::Vector(T *begin, T *end) : m_vect(begin, end) {}

template <typename T>
Vector<T>::Vector(const Vector<T>::iterator &begin, const Vector<T>::iterator &end) : m_vect(begin, end) {}

#endif // VECTOR_H
