#ifndef FREESURFACE_SIMPLELAGRANGE_H
#define FREESURFACE_SIMPLELAGRANGE_H
#include <cmath>

template<typename T>
static inline T getLagrange(T x, const std::array<T, 4>& arr)
{
    int i0 = 0;
    int i1 = 1;
    int i2 = 2;
    int i3 = 3;

    T x0 = i0;
    T x1 = i1;
    T x2 = i2;
    T x3 = i3;

    T f0 = arr[i0];
    T f1 = arr[i1];
    T f2 = arr[i2];
    T f3 = arr[i3];

    T L0 = ((x - x1) * (x - x2) * (x - x3)) * (-1./6.);
    T L1 = ((x - x0) * (x - x2) * (x - x3)) * (1./2.);
    T L2 = ((x - x0) * (x - x1) * (x - x3)) * (-1./2.);
    T L3 = ((x - x0) * (x - x1) * (x - x2)) * (1./6.);

    return f0 * L0 + f1 * L1 + f2 * L2 + f3 * L3;
}

#endif //FREESURFACE_SIMPLELAGRANGE_H
