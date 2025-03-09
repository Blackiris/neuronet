#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <iostream>

template <typename T>
class Vector
{
public:
    Vector();
    explicit Vector(const int &size);
    Vector(const int &size, T default_value);
    Vector(std::vector<T> vect);
    Vector(std::initializer_list<T> init);

    Vector operator+(const Vector& other) {
        Vector res(this->m_vect);
        for (int i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] + other.m_vect[i];
        }
        return res;
    }

    Vector operator-(const Vector& other) {
        Vector res(this->m_vect);
        for (int i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] - other.m_vect[i];
        }
        return res;
    }

    Vector operator*(const float& value) {
        Vector res(this->m_vect);
        for (int i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] * value;
        }
        return res;
    }

    Vector operator*(const int& value) {
        Vector res(this->m_vect);
        for (int i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] * value;
        }
        return res;
    }

    Vector operator/(const int& value) {
        Vector res(this->m_vect);
        for (int i=0; i<m_vect.size(); i++) {
            res.m_vect[i] = res.m_vect[i] / value;
        }
        return res;
    }



    T& operator[](const int& pos) {
        return this->m_vect[pos];
    }

    int size() const {
        return m_vect.size();
    }

    void reserve(const int& size) {
        this->m_vect.reserve(size);
    }

    T dot(const Vector<T>& other) {
        T s = 0;
        for (size_t i = 0; i<other.m_vect.size(); i++) {
            s += this->m_vect[i] * other.m_vect[i];
        }
        return s;
    }

    void normalize() {
        T sum = 0;
        for (auto& val : this->m_vect) {
            sum += val;
        }
        *this = (*this/sum);
    }

    void push(T value) {
        this->m_vect.emplace_back(value);
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

private:
    std::vector<T> m_vect;
};

template <typename T>
Vector<T>::Vector() {}

template <typename T>
Vector<T>::Vector(const int &size) {
    m_vect.assign(size, 0.f);
}

template <typename T>
Vector<T>::Vector(const int &size, T default_value) {
    m_vect.assign(size, default_value);
}

template <typename T>
Vector<T>::Vector(std::vector<T> vect): m_vect(vect) {}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> init) {
    m_vect.assign(init.begin(), init.end());
}

#endif // VECTOR_H
