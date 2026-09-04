// Vendored from code/native/shared/optimized_dsp.hpp at c26f96b; regenerate with repro/export_rack_config.py.
#pragma once
// Shared AVX2/FMA and AArch64 NEON kernels for native implementations.
//
// Keep these primitives in one place so ADAA, conventional OS, Hermite,
// AA-IIR postfilters, and future fair-filter permutations are timed using the
// same optimized FIR machinery rather than family-specific reimplementations.

#include "simd_compat.hpp"

namespace aadsp {

#if defined(AADSP_NEON) && defined(AADSP_LANE_FIR)
// Eight-output accumulator pair for the NEON lane-FMA symmetric-fold kernels.
// Four symmetric tap pairs share one coefficient vector via vfmaq_laneq_f32,
// removing the per-tap scalar splat the generic AVX2-shaped loop requires.
// The arithmetic (pair add, then FMA, ascending taps) matches the generic
// kernels exactly, so results are bit-identical.
struct NeonAccPair {
    float32x4_t lo;
    float32x4_t hi;
};

static inline NeonAccPair firsym_pairs_neon(NeonAccPair acc,
                                            const float* __restrict b,
                                            const float* __restrict c,
                                            int K, int h) {
    int i = 0;
    for (; i + 4 <= h; i += 4) {
        const float32x4_t cv = vld1q_f32(c + i);
        acc.lo = vfmaq_laneq_f32(acc.lo,
            vaddq_f32(vld1q_f32(b + i), vld1q_f32(b + K - 1 - i)), cv, 0);
        acc.hi = vfmaq_laneq_f32(acc.hi,
            vaddq_f32(vld1q_f32(b + i + 4), vld1q_f32(b + K + 3 - i)), cv, 0);
        acc.lo = vfmaq_laneq_f32(acc.lo,
            vaddq_f32(vld1q_f32(b + i + 1), vld1q_f32(b + K - 2 - i)), cv, 1);
        acc.hi = vfmaq_laneq_f32(acc.hi,
            vaddq_f32(vld1q_f32(b + i + 5), vld1q_f32(b + K + 2 - i)), cv, 1);
        acc.lo = vfmaq_laneq_f32(acc.lo,
            vaddq_f32(vld1q_f32(b + i + 2), vld1q_f32(b + K - 3 - i)), cv, 2);
        acc.hi = vfmaq_laneq_f32(acc.hi,
            vaddq_f32(vld1q_f32(b + i + 6), vld1q_f32(b + K + 1 - i)), cv, 2);
        acc.lo = vfmaq_laneq_f32(acc.lo,
            vaddq_f32(vld1q_f32(b + i + 3), vld1q_f32(b + K - 4 - i)), cv, 3);
        acc.hi = vfmaq_laneq_f32(acc.hi,
            vaddq_f32(vld1q_f32(b + i + 7), vld1q_f32(b + K - i)), cv, 3);
    }
    for (; i < h; ++i) {
        const float32x4_t cv = vdupq_n_f32(c[i]);
        acc.lo = vfmaq_f32(acc.lo,
            vaddq_f32(vld1q_f32(b + i), vld1q_f32(b + K - 1 - i)), cv);
        acc.hi = vfmaq_f32(acc.hi,
            vaddq_f32(vld1q_f32(b + i + 4), vld1q_f32(b + K + 3 - i)), cv);
    }
    if (K & 1) {
        const float32x4_t cv = vdupq_n_f32(c[h]);
        acc.lo = vfmaq_f32(acc.lo, vld1q_f32(b + h), cv);
        acc.hi = vfmaq_f32(acc.hi, vld1q_f32(b + h + 4), cv);
    }
    return acc;
}
#endif

static inline float sum_block(const float* x, int N) {
    __m256 accumulator = _mm256_setzero_ps();
    int n = 0;
    for (; n + 8 <= N; n += 8)
        accumulator = _mm256_add_ps(accumulator, _mm256_loadu_ps(x + n));
    alignas(32) float lanes[8];
    _mm256_storeu_ps(lanes, accumulator);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3]
              + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; n < N; ++n) sum += x[n];
    return sum;
}

static inline void fir_block(float* __restrict out,
                             const float* __restrict x,
                             const float* __restrict c,
                             int K, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 a = _mm256_setzero_ps();
        const float* b = x + n - K + 1;
        for (int i = 0; i < K; ++i)
            a = _mm256_fmadd_ps(_mm256_loadu_ps(b + i), _mm256_set1_ps(c[i]), a);
        _mm256_storeu_ps(out + n, a);
    }
    for (; n < N; ++n) {
        float a = 0.0f;
        const float* b = x + n - K + 1;
        for (int i = 0; i < K; ++i) a += b[i] * c[i];
        out[n] = a;
    }
}

static inline void fir_accum(float* __restrict out,
                             const float* __restrict x,
                             const float* __restrict c,
                             int K, int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 a = _mm256_loadu_ps(out + n);
        const float* b = x + n - K + 1;
        for (int i = 0; i < K; ++i)
            a = _mm256_fmadd_ps(_mm256_loadu_ps(b + i), _mm256_set1_ps(c[i]), a);
        _mm256_storeu_ps(out + n, a);
    }
    for (; n < N; ++n) {
        float a = out[n];
        const float* b = x + n - K + 1;
        for (int i = 0; i < K; ++i) a += b[i] * c[i];
        out[n] = a;
    }
}

static inline void firsym_block(float* __restrict out,
                                const float* __restrict x,
                                const float* __restrict c,
                                int K, int N) {
    const int h = K / 2;
    int n = 0;
#if defined(AADSP_NEON) && defined(AADSP_LANE_FIR)
    for (; n + 8 <= N; n += 8) {
        NeonAccPair acc{vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        acc = firsym_pairs_neon(acc, x + n - K + 1, c, K, h);
        vst1q_f32(out + n, acc.lo);
        vst1q_f32(out + n + 4, acc.hi);
    }
#else
    // Two independent accumulator chains: the single-chain form is bound by
    // FMA latency, not throughput, on both Zen 3 and Firestorm (measured
    // ~20-30% at the long-FIR tap counts by kernel_strategy_micro).
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
        _mm256_storeu_ps(out + n, _mm256_add_ps(a0, a1));
    }
#endif
    for (; n < N; ++n) {
        const float* b = x + n - K + 1;
        float a = 0.0f;
        for (int i = 0; i < h; ++i) a += (b[i] + b[K - 1 - i]) * c[i];
        if (K & 1) a += b[h] * c[h];
        out[n] = a;
    }
}

static inline void firsym_accum(float* __restrict out,
                                const float* __restrict x,
                                const float* __restrict c,
                                int K, int N) {
    const int h = K / 2;
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 a = _mm256_loadu_ps(out + n);
        const float* b = x + n - K + 1;
        for (int i = 0; i < h; ++i) {
            const __m256 v = _mm256_add_ps(_mm256_loadu_ps(b + i),
                                           _mm256_loadu_ps(b + K - 1 - i));
            a = _mm256_fmadd_ps(v, _mm256_set1_ps(c[i]), a);
        }
        if (K & 1)
            a = _mm256_fmadd_ps(_mm256_loadu_ps(b + h), _mm256_set1_ps(c[h]), a);
        _mm256_storeu_ps(out + n, a);
    }
    for (; n < N; ++n) {
        const float* b = x + n - K + 1;
        float a = out[n];
        for (int i = 0; i < h; ++i) a += (b[i] + b[K - 1 - i]) * c[i];
        if (K & 1) a += b[h] * c[h];
        out[n] = a;
    }
}

// Compute two symmetric FIRs over the same input stream in one pass. Both
// filters read windows ending at x[n], so the unrolled bodies share input
// loads via common-subexpression elimination; each filter keeps its own
// two-chain accumulation, so results are bit-identical to two separate
// firsym_block calls.
static inline void firsym_pair_block(float* __restrict out0,
                                     float* __restrict out1,
                                     const float* __restrict x,
                                     const float* __restrict c0, int K0,
                                     const float* __restrict c1, int K1,
                                     int N) {
    const int h0 = K0 / 2;
    const int h1 = K1 / 2;
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps();
        __m256 b1 = _mm256_setzero_ps();
        const float* base0 = x + n - K0 + 1;
        const float* base1 = x + n - K1 + 1;
        int i = 0;
        for (; i + 2 <= h0 && i + 2 <= h1; i += 2) {
            const __m256 u0 = _mm256_add_ps(_mm256_loadu_ps(base0 + i),
                                            _mm256_loadu_ps(base0 + K0 - 1 - i));
            a0 = _mm256_fmadd_ps(u0, _mm256_set1_ps(c0[i]), a0);
            const __m256 u1 = _mm256_add_ps(_mm256_loadu_ps(base0 + i + 1),
                                            _mm256_loadu_ps(base0 + K0 - 2 - i));
            a1 = _mm256_fmadd_ps(u1, _mm256_set1_ps(c0[i + 1]), a1);
            const __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(base1 + i),
                                            _mm256_loadu_ps(base1 + K1 - 1 - i));
            b0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(c1[i]), b0);
            const __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(base1 + i + 1),
                                            _mm256_loadu_ps(base1 + K1 - 2 - i));
            b1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(c1[i + 1]), b1);
        }
        int i0 = i;
        for (; i0 + 2 <= h0; i0 += 2) {
            const __m256 u0 = _mm256_add_ps(_mm256_loadu_ps(base0 + i0),
                                            _mm256_loadu_ps(base0 + K0 - 1 - i0));
            a0 = _mm256_fmadd_ps(u0, _mm256_set1_ps(c0[i0]), a0);
            const __m256 u1 = _mm256_add_ps(_mm256_loadu_ps(base0 + i0 + 1),
                                            _mm256_loadu_ps(base0 + K0 - 2 - i0));
            a1 = _mm256_fmadd_ps(u1, _mm256_set1_ps(c0[i0 + 1]), a1);
        }
        int i1 = i;
        for (; i1 + 2 <= h1; i1 += 2) {
            const __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(base1 + i1),
                                            _mm256_loadu_ps(base1 + K1 - 1 - i1));
            b0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(c1[i1]), b0);
            const __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(base1 + i1 + 1),
                                            _mm256_loadu_ps(base1 + K1 - 2 - i1));
            b1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(c1[i1 + 1]), b1);
        }
        if (i0 < h0) {
            const __m256 u = _mm256_add_ps(_mm256_loadu_ps(base0 + i0),
                                           _mm256_loadu_ps(base0 + K0 - 1 - i0));
            a0 = _mm256_fmadd_ps(u, _mm256_set1_ps(c0[i0]), a0);
        }
        if (i1 < h1) {
            const __m256 v = _mm256_add_ps(_mm256_loadu_ps(base1 + i1),
                                           _mm256_loadu_ps(base1 + K1 - 1 - i1));
            b0 = _mm256_fmadd_ps(v, _mm256_set1_ps(c1[i1]), b0);
        }
        if (K0 & 1)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(base0 + h0),
                                 _mm256_set1_ps(c0[h0]), a0);
        if (K1 & 1)
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(base1 + h1),
                                 _mm256_set1_ps(c1[h1]), b0);
        _mm256_storeu_ps(out0 + n, _mm256_add_ps(a0, a1));
        _mm256_storeu_ps(out1 + n, _mm256_add_ps(b0, b1));
    }
    for (; n < N; ++n) {
        const float* base0 = x + n - K0 + 1;
        const float* base1 = x + n - K1 + 1;
        float a = 0.0f;
        float b = 0.0f;
        for (int i = 0; i < h0; ++i) a += (base0[i] + base0[K0 - 1 - i]) * c0[i];
        if (K0 & 1) a += base0[h0] * c0[h0];
        for (int i = 0; i < h1; ++i) b += (base1[i] + base1[K1 - 1 - i]) * c1[i];
        if (K1 & 1) b += base1[h1] * c1[h1];
        out0[n] = a;
        out1[n] = b;
    }
}

static inline void firsym_add_delayed_block(float* __restrict out,
                                            const float* __restrict fir_input,
                                            const float* __restrict direct_input,
                                            const float* __restrict c,
                                            int K, float direct_gain,
                                            int direct_shift, int N) {
    const int h = K / 2;
    int n = 0;
#if defined(AADSP_NEON) && defined(AADSP_LANE_FIR)
    const float32x4_t gain4 = vdupq_n_f32(direct_gain);
    for (; n + 8 <= N; n += 8) {
        const float* d = direct_input + n - direct_shift;
        NeonAccPair acc{vmulq_f32(vld1q_f32(d), gain4),
                        vmulq_f32(vld1q_f32(d + 4), gain4)};
        acc = firsym_pairs_neon(acc, fir_input + n - K + 1, c, K, h);
        vst1q_f32(out + n, acc.lo);
        vst1q_f32(out + n + 4, acc.hi);
    }
#else
    const __m256 gain = _mm256_set1_ps(direct_gain);
    for (; n + 8 <= N; n += 8) {
        __m256 a0 = _mm256_mul_ps(
            _mm256_loadu_ps(direct_input + n - direct_shift), gain);
        __m256 a1 = _mm256_setzero_ps();
        const float* b = fir_input + n - K + 1;
        int i = 0;
        for (; i + 2 <= h; i += 2) {
            const __m256 v0 = _mm256_add_ps(
                _mm256_loadu_ps(b + i), _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(c[i]), a0);
            const __m256 v1 = _mm256_add_ps(
                _mm256_loadu_ps(b + i + 1), _mm256_loadu_ps(b + K - 2 - i));
            a1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(c[i + 1]), a1);
        }
        if (i < h) {
            const __m256 v = _mm256_add_ps(
                _mm256_loadu_ps(b + i), _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v, _mm256_set1_ps(c[i]), a0);
        }
        if (K & 1)
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(b + h), _mm256_set1_ps(c[h]), a0);
        _mm256_storeu_ps(out + n, _mm256_add_ps(a0, a1));
    }
#endif
    for (; n < N; ++n) {
        const float* b = fir_input + n - K + 1;
        float acc = direct_gain * direct_input[n - direct_shift];
        for (int i = 0; i < h; ++i) acc += (b[i] + b[K - 1 - i]) * c[i];
        if (K & 1) acc += b[h] * c[h];
        out[n] = acc;
    }
}

static inline void fircross_block(float* __restrict out_forward,
                                  float* __restrict out_reverse,
                                  const float* __restrict x,
                                  const float* __restrict c,
                                  int K, int N) {
    const int h = K / 2;
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 forward = _mm256_setzero_ps();
        __m256 reverse = _mm256_setzero_ps();
        const float* b = x + n - K + 1;
        for (int i = 0; i < h; ++i) {
            const __m256 left = _mm256_loadu_ps(b + i);
            const __m256 right = _mm256_loadu_ps(b + K - 1 - i);
            const __m256 sum = _mm256_add_ps(left, right);
            const __m256 diff = _mm256_sub_ps(left, right);
            const float coeff_sum = 0.5f * (c[i] + c[K - 1 - i]);
            const float coeff_diff = 0.5f * (c[i] - c[K - 1 - i]);
            const __m256 even = _mm256_mul_ps(sum, _mm256_set1_ps(coeff_sum));
            const __m256 odd = _mm256_mul_ps(diff, _mm256_set1_ps(coeff_diff));
            forward = _mm256_add_ps(forward, _mm256_add_ps(even, odd));
            reverse = _mm256_add_ps(reverse, _mm256_sub_ps(even, odd));
        }
        if (K & 1) {
            const __m256 center = _mm256_mul_ps(
                _mm256_loadu_ps(b + h), _mm256_set1_ps(c[h]));
            forward = _mm256_add_ps(forward, center);
            reverse = _mm256_add_ps(reverse, center);
        }
        _mm256_storeu_ps(out_forward + n, forward);
        _mm256_storeu_ps(out_reverse + n, reverse);
    }
    for (; n < N; ++n) {
        const float* b = x + n - K + 1;
        float forward = 0.0f;
        float reverse = 0.0f;
        for (int i = 0; i < h; ++i) {
            const float sum = b[i] + b[K - 1 - i];
            const float diff = b[i] - b[K - 1 - i];
            const float coeff_sum = 0.5f * (c[i] + c[K - 1 - i]);
            const float coeff_diff = 0.5f * (c[i] - c[K - 1 - i]);
            const float even = sum * coeff_sum;
            const float odd = diff * coeff_diff;
            forward += even + odd;
            reverse += even - odd;
        }
        if (K & 1) {
            const float center = b[h] * c[h];
            forward += center;
            reverse += center;
        }
        out_forward[n] = forward;
        out_reverse[n] = reverse;
    }
}

static inline void fircross_sum(float* __restrict out,
                                const float* __restrict forward,
                                const float* __restrict reverse,
                                const float* __restrict c,
                                int K, int N, bool clear) {
    const int h = K / 2;
    int n = 0;
    for (; n + 8 <= N; n += 8) {
        __m256 acc = clear ? _mm256_setzero_ps() : _mm256_loadu_ps(out + n);
        const float* a = forward + n - K + 1;
        const float* b = reverse + n - K + 1;
        for (int i = 0; i < h; ++i) {
            const __m256 first = _mm256_add_ps(
                _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + K - 1 - i));
            const __m256 second = _mm256_add_ps(
                _mm256_loadu_ps(a + K - 1 - i), _mm256_loadu_ps(b + i));
            acc = _mm256_fmadd_ps(first, _mm256_set1_ps(c[i]), acc);
            acc = _mm256_fmadd_ps(second, _mm256_set1_ps(c[K - 1 - i]), acc);
        }
        if (K & 1) {
            const __m256 center = _mm256_add_ps(
                _mm256_loadu_ps(a + h), _mm256_loadu_ps(b + h));
            acc = _mm256_fmadd_ps(center, _mm256_set1_ps(c[h]), acc);
        }
        _mm256_storeu_ps(out + n, acc);
    }
    for (; n < N; ++n) {
        const float* a = forward + n - K + 1;
        const float* b = reverse + n - K + 1;
        float acc = clear ? 0.0f : out[n];
        for (int i = 0; i < h; ++i) {
            acc += (a[i] + b[K - 1 - i]) * c[i]
                 + (a[K - 1 - i] + b[i]) * c[K - 1 - i];
        }
        if (K & 1) acc += (a[h] + b[h]) * c[h];
        out[n] = acc;
    }
}

static inline __m256 smoothstep_d7(__m256 x) {
    const __m256 c1 = _mm256_set1_ps(2.1875f);
    const __m256 c3 = _mm256_set1_ps(-2.1875f);
    const __m256 c5 = _mm256_set1_ps(1.3125f);
    const __m256 c7 = _mm256_set1_ps(-0.3125f);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 a = _mm256_fmadd_ps(c7, x2, c5);
    a = _mm256_fmadd_ps(a, x2, c3);
    a = _mm256_fmadd_ps(a, x2, c1);
    return _mm256_mul_ps(x, a);
}

static inline float smoothstep_d7(float x) {
    const float x2 = x * x;
    return x * (2.1875f + x2 * (-2.1875f + x2 * (1.3125f - 0.3125f * x2)));
}

static inline void smoothstep_d7_block(float* __restrict out,
                                       const float* __restrict x,
                                       int N) {
    int n = 0;
    for (; n + 8 <= N; n += 8)
        _mm256_storeu_ps(out + n, smoothstep_d7(_mm256_loadu_ps(x + n)));
    for (; n < N; ++n) out[n] = smoothstep_d7(x[n]);
}

static inline void firsym_smoothstep_d7_block(float* __restrict out,
                                              const float* __restrict x,
                                              const float* __restrict c,
                                              int K, int N) {
    const int h = K / 2;
    int n = 0;
#if defined(AADSP_NEON) && defined(AADSP_LANE_FIR)
    for (; n + 8 <= N; n += 8) {
        NeonAccPair acc{vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        acc = firsym_pairs_neon(acc, x + n - K + 1, c, K, h);
        const __m256 shaped = smoothstep_d7(__m256{acc.lo, acc.hi});
        _mm256_storeu_ps(out + n, shaped);
    }
#else
    for (; n + 8 <= N; n += 8) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        const float* b = x + n - K + 1;
        int i = 0;
        for (; i + 2 <= h; i += 2) {
            const __m256 v0 = _mm256_add_ps(
                _mm256_loadu_ps(b + i), _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v0, _mm256_set1_ps(c[i]), a0);
            const __m256 v1 = _mm256_add_ps(
                _mm256_loadu_ps(b + i + 1), _mm256_loadu_ps(b + K - 2 - i));
            a1 = _mm256_fmadd_ps(v1, _mm256_set1_ps(c[i + 1]), a1);
        }
        if (i < h) {
            const __m256 v = _mm256_add_ps(
                _mm256_loadu_ps(b + i), _mm256_loadu_ps(b + K - 1 - i));
            a0 = _mm256_fmadd_ps(v, _mm256_set1_ps(c[i]), a0);
        }
        if (K & 1)
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(b + h), _mm256_set1_ps(c[h]), a0);
        _mm256_storeu_ps(out + n, smoothstep_d7(_mm256_add_ps(a0, a1)));
    }
#endif
    for (; n < N; ++n) {
        const float* b = x + n - K + 1;
        float acc = 0.0f;
        for (int i = 0; i < h; ++i) acc += (b[i] + b[K - 1 - i]) * c[i];
        if (K & 1) acc += b[h] * c[h];
        out[n] = smoothstep_d7(acc);
    }
}

} // namespace aadsp
