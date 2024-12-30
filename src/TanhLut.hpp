#include <array>
#include <cmath>
#include <algorithm>
#include "plugin.hpp"

#ifndef FREESURFACE_TANHLUT_HPP
#define FREESURFACE_TANHLUT_HPP
template <int N, bool UseLagrange = false, typename T = float>
struct TanhLUT
{
    static_assert(N > 1, "LUT size must be greater than 1.");

    // Public members
    T minX;
    T maxX;
    T step;
    T clampMax;
    T clampMin;
    const T L0_Base;
    const T L1_Base;
    const T L2_Base;
    const T L3_Base;
    std::array<T, N> values;
    std::array<T, N> integralValues; // ln(cosh(x)) values

    // Constructor: builds the LUT for tanh(x) from minX to maxX
    TanhLUT(T min_x, T max_x) :
            minX(min_x),
            maxX(max_x),
            step((maxX - minX) / (N - 1)),
            L0_Base(-1./(6.*step*step*step)),
            L1_Base(1./(2.*step*step*step)),
            L2_Base(-1./(2.*step*step*step)),
            L3_Base(1./(6.*step*step*step)),
            clampMin(minX + step),
            clampMax(maxX - 3. * step)

    {
        for (int i = 0; i < N; ++i)
        {
            T x = minX + i * step;
            values[i] = std::tanh(x);
            integralValues[i] = std::log(std::cosh(x));
        }
    }

    // Get value at x
    inline T get(T x) const {
        x = clamp(x, clampMin, clampMax);
        if constexpr (UseLagrange) {
            return getLagrange(x, values);
        } else {
            return getLinear(x, values);
        }
    }

    inline T getIntegral(T x) const {
        x = clamp(x, clampMin, clampMax);
        if constexpr (UseLagrange) {
            return getLagrange(x, integralValues);
        } else {
            return getLinear(x, integralValues);
        }
    }

    inline T averageOverInterval(T begin, T end) const {
        // Ensure begin < end, if not swap
        if (begin > end) std::swap(begin, end);

        T integralBegin = getIntegral(begin);
        T integralEnd   = getIntegral(end);
        T integralDiff  = integralEnd - integralBegin;
        T length = end - begin;

        // machine epsilon is too low here
        if (abs(length) <= 1.e-6) {
            // Degenerate interval, just return tanh
            return get(begin);
        }

        return integralDiff / length;
    }

private:
    inline T getLinear(T x, const std::array<T, N>& arr) const
    {
        T t = (x - minX) / step;
        int i = static_cast<int>(t);
        T frac = t - i;
        T y0 = arr[i];
        T y1 = arr[i + 1];
        return y0 + (y1 - y0) * frac;
    }

    inline T getLagrange(T x, const std::array<T, N>& arr) const
    {
        T t = (x - minX) / step;
        int i = static_cast<int>(std::floor(t));

        int i0 = i - 1;
        int i1 = i;
        int i2 = i + 1;
        int i3 = i + 2;

        T x0 = minX + i0 * step;
        T x1 = minX + i1 * step;
        T x2 = minX + i2 * step;
        T x3 = minX + i3 * step;

        T f0 = arr[i0];
        T f1 = arr[i1];
        T f2 = arr[i2];
        T f3 = arr[i3];

        T L0 = ((x - x1) * (x - x2) * (x - x3)) * L0_Base;
        T L1 = ((x - x0) * (x - x2) * (x - x3)) * L1_Base;
        T L2 = ((x - x0) * (x - x1) * (x - x3)) * L2_Base;
        T L3 = ((x - x0) * (x - x1) * (x - x2)) * L3_Base;

        return f0 * L0 + f1 * L1 + f2 * L2 + f3 * L3;
    }
};

template <int N>
struct TanhLUT4
{
    static_assert(N > 1, "LUT size must be greater than 1.");

    // Public members
    const float minX;
    const float maxX;
    const float step;
    const float L0_Base;
    const float L1_Base;
    const float L2_Base;
    const float L3_Base;
    const float clampMin;
    const float clampMax;

    std::array<float, N> values;
    std::array<float, N> integralValues; // ln(cosh(x)) values

    // Constructor: builds the LUT for tanh(x) from minX to maxX
    TanhLUT4(float min_x, float max_x, float sc) :
            minX(min_x),
            maxX(max_x),
            step((maxX - minX) / (N - 1)),
            L0_Base(-1./(6.*step*step*step)),
            L1_Base(1./(2.*step*step*step)),
            L2_Base(-1./(2.*step*step*step)),
            L3_Base(1./(6.*step*step*step)),
            clampMin(minX + step),
            clampMax(maxX - 3. * step)
    {
        for (int i = 0; i < N; ++i)
        {
            float x = minX + i * step;
            values[i] = sc*std::tanh(x/sc);
            integralValues[i] = sc*sc*std::log(std::cosh(x/sc));
        }
    }

    // Get value at x
    inline float_4 get(float_4 x) const {
        x = simd::clamp(x, clampMin, clampMax);
        return getLinear(x, values);
    }

    inline float_4 getIntegral(float_4 x) const {
        x = simd::clamp(x, clampMin, clampMax);
        return getLinear(x, integralValues);
    }

    inline float_4 averageOverInterval(float_4 begin, float_4 grad) const {

        float_4 end = begin + grad;
        float_4 t_begin = simd::ifelse(begin > end, end, begin);
        end = simd::ifelse(begin > end, begin, end);
        begin = t_begin;

        float_4 integralBegin = getIntegral(begin);
        float_4 integralEnd   = getIntegral(end);
        float_4 integralDiff  = integralEnd - integralBegin;
        float_4 length = end - begin;

        // TODO: this is bad, does an extra getLinear for the whole float_4
        return simd::ifelse(abs(length) <= 1.e-6, get(begin), integralDiff / length);
    }

private:

    inline float_4 getLinear(float_4 x, const std::array<float, N>& arr) const {
        float_4 t = (x - minX) / step;
        int32_4 i = int32_4(t);
        float_4 frac = t - float_4(i);

        float_4 y0 = float_4(arr[i[0]], arr[i[1]], arr[i[2]], arr[i[3]]);
        float_4 y1 = float_4(arr[i[0]+1], arr[i[1]+1], arr[i[2]+1], arr[i[3]+1]);
        return y0 + (y1 - y0) * frac;
    }
};
#endif //FREESURFACE_TANHLUT_HPP
