// Vendored from code/native/shared/adaa_d7.hpp at c26f96b; regenerate with repro/export_rack_config.py.
#pragma once
// Shared exact polynomial ADAA kernels for the degree-7 smoothstep target.
//
// The implementation uses complete homogeneous symmetric polynomials rather
// than divided-difference denominators, so repeated/near-repeated samples need
// no exceptional branch.  This is the same exact polynomial specialization
// used by the Python quality benchmark, expressed as AVX2/FMA primitives for
// native timing work. The vector operations map to AVX2/FMA or AArch64 NEON.

#include "simd_compat.hpp"

namespace aadsp {

constexpr int binom_constexpr(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    int r = 1;
    for (int i = 1; i <= k; ++i) r = (r * (n - k + i)) / i;
    return r;
}

template<int P>
static inline __m256 adaa_d7_combine(const __m256 (&h)[8]) {
    __m256 y = _mm256_mul_ps(
        h[1], _mm256_set1_ps(2.1875f / float(binom_constexpr(1 + P, P))));
    y = _mm256_fmadd_ps(
        h[3], _mm256_set1_ps(-2.1875f / float(binom_constexpr(3 + P, P))), y);
    y = _mm256_fmadd_ps(
        h[5], _mm256_set1_ps(1.3125f / float(binom_constexpr(5 + P, P))), y);
    y = _mm256_fmadd_ps(
        h[7], _mm256_set1_ps(-0.3125f / float(binom_constexpr(7 + P, P))), y);
    return y;
}

template<int P>
static inline __m256 adaa_d7_vec(const float* x, int n) {
    static_assert(P >= 1, "ADAA order must be >= 1");
    __m256 h[8];
    h[0] = _mm256_set1_ps(1.0f);
    for (int k = 1; k <= 7; ++k) h[k] = _mm256_setzero_ps();

    for (int r = 0; r <= P; ++r) {
        const __m256 v = _mm256_loadu_ps(x + n - r);
        // Ascending k implements multiplication by 1/(1-v t), allowing
        // repeated powers/nodes exactly without divided-difference branches.
        for (int k = 1; k <= 7; ++k)
            h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
    }
    return adaa_d7_combine<P>(h);
}

template<int P>
static inline float adaa_d7_scalar(const float* x, int n) {
    static_assert(P >= 1, "ADAA order must be >= 1");
    double h[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r <= P; ++r) {
        const double v = x[n - r];
        for (int k = 1; k <= 7; ++k) h[k] += v * h[k - 1];
    }
    return float(
        2.1875 * h[1] / double(binom_constexpr(1 + P, P))
      - 2.1875 * h[3] / double(binom_constexpr(3 + P, P))
      + 1.3125 * h[5] / double(binom_constexpr(5 + P, P))
      - 0.3125 * h[7] / double(binom_constexpr(7 + P, P)));
}

template<int P>
static inline void adaa_d7_block(float* __restrict out,
                                  const float* __restrict x,
                                  int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, adaa_d7_vec<P>(x, n));
    for (; n < N; ++n) out[n] = adaa_d7_scalar<P>(x, n);
}

// Phase-pair ADAA: evaluate the same rate-2R interleaved ADAA operator
// directly on the two deinterleaved polyphase buffers, avoiding the
// interleave/deinterleave round trip.  For the stream x[2n] = self[n],
// x[2n+1] = other[n] the window sample at lag r lives at
//   self[n - r/2]                     for even r,
//   other[n - (r+1)/2 + SELF_ODD]     for odd r,
// where SELF_ODD selects whether `self` is the odd phase.  The factors are
// consumed in the same lag order as the interleaved kernel, so results are
// bit-identical to adaa_d7_block on the interleaved stream.  Each buffer must
// provide (P+1)/2 samples of history before index 0.
template<int P, int SELF_ODD>
static inline __m256 adaa_d7_vec_phase(const float* self, const float* other,
                                       int n) {
    static_assert(P >= 1, "ADAA order must be >= 1");
    static_assert(SELF_ODD == 0 || SELF_ODD == 1, "SELF_ODD selects the phase");
    __m256 h[8];
    h[0] = _mm256_set1_ps(1.0f);
    for (int k = 1; k <= 7; ++k) h[k] = _mm256_setzero_ps();

    for (int r = 0; r <= P; ++r) {
        const float* source = (r & 1) ? other : self;
        const int offset = (r & 1) ? ((r + 1) / 2 - SELF_ODD) : (r / 2);
        const __m256 v = _mm256_loadu_ps(source + n - offset);
        for (int k = 1; k <= 7; ++k)
            h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
    }
    return adaa_d7_combine<P>(h);
}

template<int P, int SELF_ODD>
static inline float adaa_d7_scalar_phase(const float* self, const float* other,
                                         int n) {
    static_assert(P >= 1, "ADAA order must be >= 1");
    double h[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r <= P; ++r) {
        const float* source = (r & 1) ? other : self;
        const int offset = (r & 1) ? ((r + 1) / 2 - SELF_ODD) : (r / 2);
        const double v = source[n - offset];
        for (int k = 1; k <= 7; ++k) h[k] += v * h[k - 1];
    }
    return float(
        2.1875 * h[1] / double(binom_constexpr(1 + P, P))
      - 2.1875 * h[3] / double(binom_constexpr(3 + P, P))
      + 1.3125 * h[5] / double(binom_constexpr(5 + P, P))
      - 0.3125 * h[7] / double(binom_constexpr(7 + P, P)));
}

template<int P, int SELF_ODD>
static inline void adaa_d7_phase_block(float* __restrict out,
                                       const float* __restrict self,
                                       const float* __restrict other,
                                       int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, adaa_d7_vec_phase<P, SELF_ODD>(self, other, n));
    for (; n < N; ++n) out[n] = adaa_d7_scalar_phase<P, SELF_ODD>(self, other, n);
}

// Arbitrary-R phase-native ADAA (M2 generalization of the two-phase kernel).
// For the interleaved stream x[R*n + p] = phases[p][n], the window sample at
// lag r for output phase PHASE lives at
//   phases[(PHASE - r) mod R][n + floor((PHASE - r) / R)].
// Lags are consumed in ascending order, matching the interleaved kernel's
// factor order, so results agree with adaa_d7_block on the interleaved
// stream. Each phase buffer must provide ceil(P / R) samples of history.
template<int P, int R, int PHASE>
static inline __m256 adaa_d7_vec_rphase(const float* const* phases, int n) {
    static_assert(P >= 1, "ADAA order must be >= 1");
    static_assert(R >= 2, "phase count must be >= 2");
    static_assert(PHASE >= 0 && PHASE < R, "PHASE must index a phase");
    __m256 h[8];
    h[0] = _mm256_set1_ps(1.0f);
    for (int k = 1; k <= 7; ++k) h[k] = _mm256_setzero_ps();

    for (int r = 0; r <= P; ++r) {
        const int q = PHASE - r;
        const int source = ((q % R) + R) % R;
        const int offset = (q - source) / R;  // floor division
        const __m256 v = _mm256_loadu_ps(phases[source] + n + offset);
        for (int k = 1; k <= 7; ++k)
            h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
    }
    return adaa_d7_combine<P>(h);
}

template<int P, int R, int PHASE>
static inline float adaa_d7_scalar_rphase(const float* const* phases, int n) {
    double h[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r <= P; ++r) {
        const int q = PHASE - r;
        const int source = ((q % R) + R) % R;
        const int offset = (q - source) / R;
        const double v = phases[source][n + offset];
        for (int k = 1; k <= 7; ++k) h[k] += v * h[k - 1];
    }
    return float(
        2.1875 * h[1] / double(binom_constexpr(1 + P, P))
      - 2.1875 * h[3] / double(binom_constexpr(3 + P, P))
      + 1.3125 * h[5] / double(binom_constexpr(5 + P, P))
      - 0.3125 * h[7] / double(binom_constexpr(7 + P, P)));
}

template<int P, int R, int PHASE>
static inline void adaa_d7_rphase_block(float* __restrict out,
                                        const float* const* phases, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, adaa_d7_vec_rphase<P, R, PHASE>(phases, n));
    for (; n < N; ++n) out[n] = adaa_d7_scalar_rphase<P, R, PHASE>(phases, n);
}

// Spectrally-flat ADAA (alias_frontier_bench.method_adaa_flat): the target's
// explicit linear term is removed from the ADAA'd polynomial (residual) and
// re-added as a direct through-path delayed by D samples. Matches the
// canonical Python semantics: residual coefficients are the d7 set with the
// degree-1 coefficient zeroed; through gain is that degree-1 coefficient.
template<int P>
static inline __m256 adaa_d7_combine_residual(const __m256 (&h)[8]) {
    __m256 y = _mm256_mul_ps(
        h[3], _mm256_set1_ps(-2.1875f / float(binom_constexpr(3 + P, P))));
    y = _mm256_fmadd_ps(
        h[5], _mm256_set1_ps(1.3125f / float(binom_constexpr(5 + P, P))), y);
    y = _mm256_fmadd_ps(
        h[7], _mm256_set1_ps(-0.3125f / float(binom_constexpr(7 + P, P))), y);
    return y;
}

template<int P, int D>
static inline void adaa_d7_flat_block(float* __restrict out,
                                      const float* __restrict x, int N) {
    static_assert(P >= 1 && D >= 0, "invalid flat-ADAA parameters");
    const __m256 linear = _mm256_set1_ps(2.1875f);
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 h[8];
        h[0] = _mm256_set1_ps(1.0f);
        for (int k = 1; k <= 7; ++k) h[k] = _mm256_setzero_ps();
        for (int r = 0; r <= P; ++r) {
            const __m256 v = _mm256_loadu_ps(x + n - r);
            for (int k = 1; k <= 7; ++k)
                h[k] = _mm256_fmadd_ps(v, h[k - 1], h[k]);
        }
        const __m256 y = _mm256_fmadd_ps(
            _mm256_loadu_ps(x + n - D), linear,
            adaa_d7_combine_residual<P>(h));
        _mm256_storeu_ps(out + n, y);
    }
    for (; n < N; ++n) {
        double h[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int r = 0; r <= P; ++r) {
            const double v = x[n - r];
            for (int k = 1; k <= 7; ++k) h[k] += v * h[k - 1];
        }
        out[n] = float(
            -2.1875 * h[3] / double(binom_constexpr(3 + P, P))
          + 1.3125 * h[5] / double(binom_constexpr(5 + P, P))
          - 0.3125 * h[7] / double(binom_constexpr(7 + P, P))
          + 2.1875 * double(x[n - D]));
    }
}

} // namespace aadsp
