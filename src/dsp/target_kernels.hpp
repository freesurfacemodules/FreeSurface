// Vendored from code/native/shared/target_kernels.hpp at c26f96b; regenerate with repro/export_rack_config.py.
#pragma once
// Target-generic polynomial and ADAA kernels (roadmap M4b).
//
// Kernels are templated on a target trait (target_traits.hpp). For
// odd_smoothstep_d7 the generated instruction sequences are identical to the
// specialized d7 kernels (odd-form Horner and the same ascending-lag
// homogeneous recurrence/combine order), so results are bit-identical and the
// d7 fast path needs no special-casing. Loops over trait coefficients use
// constexpr bounds/values and fully unroll; zero coefficients drop out at
// compile time.
#include "adaa_d7.hpp"  // binom_constexpr + SIMD compat
#include "target_traits.hpp"

namespace aadsp {

template<class T>
static inline __m256 poly_eval(__m256 x) {
    if constexpr (T::odd_only) {
        const __m256 x2 = _mm256_mul_ps(x, x);
        constexpr int half = T::degree / 2;  // highest odd index = 2*half+1
        __m256 acc = _mm256_set1_ps(T::coeffs[2 * half + 1]);
        for (int k = half - 1; k >= 0; --k)
            acc = _mm256_fmadd_ps(acc, x2,
                                  _mm256_set1_ps(T::coeffs[2 * k + 1]));
        return _mm256_mul_ps(x, acc);
    } else if constexpr (T::even_only) {
        const __m256 x2 = _mm256_mul_ps(x, x);
        constexpr int half = T::degree / 2;  // highest even index = 2*half
        __m256 acc = _mm256_set1_ps(T::coeffs[2 * half]);
        for (int k = half - 1; k >= 0; --k)
            acc = _mm256_fmadd_ps(acc, x2, _mm256_set1_ps(T::coeffs[2 * k]));
        return acc;
    } else {
        __m256 acc = _mm256_set1_ps(T::coeffs[T::degree]);
        for (int k = T::degree - 1; k >= 0; --k)
            acc = _mm256_fmadd_ps(acc, x, _mm256_set1_ps(T::coeffs[k]));
        return acc;
    }
}

template<class T>
static inline float poly_eval(float x) {
    if constexpr (T::odd_only) {
        const float x2 = x * x;
        constexpr int half = T::degree / 2;
        float acc = T::coeffs[2 * half + 1];
        for (int k = half - 1; k >= 0; --k)
            acc = acc * x2 + T::coeffs[2 * k + 1];
        return x * acc;
    } else if constexpr (T::even_only) {
        const float x2 = x * x;
        constexpr int half = T::degree / 2;
        float acc = T::coeffs[2 * half];
        for (int k = half - 1; k >= 0; --k)
            acc = acc * x2 + T::coeffs[2 * k];
        return acc;
    } else {
        float acc = T::coeffs[T::degree];
        for (int k = T::degree - 1; k >= 0; --k)
            acc = acc * x + T::coeffs[k];
        return acc;
    }
}

// Clamped evaluation: the input is limited to [-T::clamp, T::clamp] before
// the polynomial, i.e. PolynomialTarget.eval semantics (a smoothstep held
// at its end values).  Opt-in (Clamp template flag on the block kernels and
// pipelines); the benchmarked lowerings and their float64 references use
// the unclamped forms above.
template<class T>
static inline __m256 poly_eval_clamped(__m256 x) {
    const __m256 hi = _mm256_set1_ps(T::clamp);
    const __m256 lo = _mm256_set1_ps(-T::clamp);
    return poly_eval<T>(_mm256_min_ps(hi, _mm256_max_ps(lo, x)));
}

template<class T>
static inline float poly_eval_clamped(float x) {
    return poly_eval<T>(x > T::clamp ? T::clamp : (x < -T::clamp ? -T::clamp : x));
}

template<class T, bool Clamp>
static inline __m256 poly_eval_sel(__m256 x) {
    if constexpr (Clamp) return poly_eval_clamped<T>(x);
    else return poly_eval<T>(x);
}

template<class T, bool Clamp>
static inline float poly_eval_sel(float x) {
    if constexpr (Clamp) return poly_eval_clamped<T>(x);
    else return poly_eval<T>(x);
}

template<class T, bool Clamp = false>
static inline void poly_block(float* __restrict out,
                              const float* __restrict x, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, poly_eval_sel<T, Clamp>(_mm256_loadu_ps(x + n)));
    for (; n < N; ++n) out[n] = poly_eval_sel<T, Clamp>(x[n]);
}

// Pointwise target minus its explicit linear term (the nonlinear residual
// used by linear-bypass topologies): y = f(x) - c1 * x.
template<class T>
static inline void poly_residual_block(float* __restrict out,
                                       const float* __restrict x, int N) {
    const __m256 linear = _mm256_set1_ps(T::coeffs[1]);
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        const __m256 v = _mm256_loadu_ps(x + n);
        _mm256_storeu_ps(out + n,
                         _mm256_fnmadd_ps(linear, v, poly_eval<T>(v)));
    }
    for (; n < N; ++n) out[n] = poly_eval<T>(x[n]) - T::coeffs[1] * x[n];
}

// Symmetric FIR fused with target evaluation (generalizes
// firsym_smoothstep_d7_block; same two-chain accumulation).
template<class T, bool Clamp = false>
static inline void firsym_poly_block(float* __restrict out,
                                     const float* __restrict x,
                                     const float* __restrict c,
                                     int K, int N) {
    const int h = K / 2;
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        const float* b = x + n - K + 1;
        int i = 0;
        for (; i + 2 <= h; i += 2) {
            const __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(b + i),
                                            _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(c[i]), a0);
            const __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(b + i + 1),
                                            _mm256_loadu_ps(b + K - 2 - i));
            a1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(c[i + 1]), a1);
        }
        if (i < h) {
            const __m256 v = _mm256_add_ps(_mm256_loadu_ps(b + i),
                                           _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v, _mm256_set1_ps(c[i]), a0);
        }
        if (K & 1)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(b + h), _mm256_set1_ps(c[h]),
                                 a0);
        _mm256_storeu_ps(out + n, poly_eval_sel<T, Clamp>(_mm256_add_ps(a0, a1)));
    }
    for (; n < N; ++n) {
        const float* b = x + n - K + 1;
        float a = 0.0f;
        for (int i = 0; i < h; ++i) a += (b[i] + b[K - 1 - i]) * c[i];
        if (K & 1) a += b[h] * c[h];
        out[n] = poly_eval_sel<T, Clamp>(a);
    }
}

// Generic exact polynomial ADAA (interleaved stream).
template<class T, int P>
static inline __m256 adaa_poly_vec(const float* x, int n) {
    static_assert(P >= 1);
    constexpr int D = T::degree;
    __m256 h[D + 1];
    h[0] = _mm256_set1_ps(1.0f);
    for (int k = 1; k <= D; ++k) h[k] = _mm256_setzero_ps();
    for (int r = 0; r <= P; ++r) {
        const __m256 v = _mm256_loadu_ps(x + n - r);
        for (int k = 1; k <= D; ++k)
            h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
    }
    __m256 y = _mm256_setzero_ps();
    for (int k = 1; k <= D; ++k)
        if (T::coeffs[k] != 0.0f)
            y = _mm256_fmadd_ps(h[k], _mm256_set1_ps(
                    T::coeffs[k] / float(binom_constexpr(k + P, P))), y);
    return y;
}

template<class T, int P>
static inline float adaa_poly_scalar(const float* x, int n) {
    constexpr int D = T::degree;
    double h[D + 1] = {};
    h[0] = 1.0;
    for (int r = 0; r <= P; ++r) {
        const double v = x[n - r];
        for (int k = 1; k <= D; ++k) h[k] += v * h[k - 1];
    }
    double y = 0.0;
    for (int k = 1; k <= D; ++k)
        if (T::coeffs[k] != 0.0f)
            y += double(T::coeffs[k]) * h[k]
                 / double(binom_constexpr(k + P, P));
    return float(y);
}

template<class T, int P>
static inline void adaa_poly_block(float* __restrict out,
                                   const float* __restrict x, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, adaa_poly_vec<T, P>(x, n));
    for (; n < N; ++n) out[n] = adaa_poly_scalar<T, P>(x, n);
}

// Generic phase-native ADAA (arbitrary R), mirroring adaa_d7_vec_rphase.
template<class T, int P, int R, int PHASE>
static inline __m256 adaa_poly_vec_rphase(const float* const* phases, int n) {
    static_assert(P >= 1 && R >= 2 && PHASE >= 0 && PHASE < R);
    constexpr int D = T::degree;
    __m256 h[D + 1];
    h[0] = _mm256_set1_ps(1.0f);
    for (int k = 1; k <= D; ++k) h[k] = _mm256_setzero_ps();
    for (int r = 0; r <= P; ++r) {
        const int q = PHASE - r;
        const int source = ((q % R) + R) % R;
        const int offset = (q - source) / R;
        const __m256 v = _mm256_loadu_ps(phases[source] + n + offset);
        for (int k = 1; k <= D; ++k)
            h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
    }
    __m256 y = _mm256_setzero_ps();
    for (int k = 1; k <= D; ++k)
        if (T::coeffs[k] != 0.0f)
            y = _mm256_fmadd_ps(h[k], _mm256_set1_ps(
                    T::coeffs[k] / float(binom_constexpr(k + P, P))), y);
    return y;
}

template<class T, int P, int R, int PHASE>
static inline float adaa_poly_scalar_rphase(const float* const* phases,
                                            int n) {
    constexpr int D = T::degree;
    double h[D + 1] = {};
    h[0] = 1.0;
    for (int r = 0; r <= P; ++r) {
        const int q = PHASE - r;
        const int source = ((q % R) + R) % R;
        const int offset = (q - source) / R;
        const double v = phases[source][n + offset];
        for (int k = 1; k <= D; ++k) h[k] += v * h[k - 1];
    }
    double y = 0.0;
    for (int k = 1; k <= D; ++k)
        if (T::coeffs[k] != 0.0f)
            y += double(T::coeffs[k]) * h[k]
                 / double(binom_constexpr(k + P, P));
    return float(y);
}

template<class T, int P, int R, int PHASE>
static inline void adaa_poly_rphase_block(float* __restrict out,
                                          const float* const* phases, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n,
                         adaa_poly_vec_rphase<T, P, R, PHASE>(phases, n));
    for (; n < N; ++n)
        out[n] = adaa_poly_scalar_rphase<T, P, R, PHASE>(phases, n);
}

} // namespace aadsp
