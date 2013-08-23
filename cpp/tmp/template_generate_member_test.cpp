#include <cmath>
#include <iostream>

// A test of using TMP to generate members of vector
using namespace std;

template<typename T, size_t SIZE>
struct _member
    : public _member<T, SIZE-1> {
    T member;

    inline T get(size_t level) {
        if ( level == SIZE ) {
            return member;
        } else {
            return _member<T, SIZE-1>::get(level);
        }
    }

    inline void set(size_t level, T value) {
        if ( level == SIZE ) {
            member = value;
        } else {
            _member<T, SIZE-1>::set(level, value);
        }
    }
};

template<typename T>
struct _member<T, 0> {
    inline T get(size_t level) {
        return T(0);
    }

    inline void set(size_t level, T value) {
        return;
    }
};

template<typename T, size_t SIZE>
struct vector {
    union {
        _member<T, SIZE> i;
        T axis[SIZE];
    };
    void print() {
        cout << SIZE << endl;
    }
};

int main() {
    vector<int, 1> vec1;
    vec1.i.set(1, 1);
    vec1.print();
    cout << vec1.i.get(1) << endl;

    vector<int, 3> vec3;
    vec3.i.set(1, 1);
    vec3.i.set(2, 2);
    vec3.i.set(3, 3);

    cout << vec3.i.get(1) << endl << vec3.i.get(2) << endl << vec3.i.get(3) << endl;
    cout << vec3.axis[0] << endl << vec3.axis[1] << endl << vec3.axis[2] << endl;
}
