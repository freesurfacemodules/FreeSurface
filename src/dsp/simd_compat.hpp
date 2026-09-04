// Vendored from code/native/shared/simd_compat.hpp at c26f96b; regenerate with repro/export_rack_config.py.
#pragma once

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

#elif defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

// Marks the availability of NEON-native kernel specializations (lane-indexed
// FMA forms that the generic AVX2-shaped code cannot express).
#define AADSP_NEON 1

namespace aadsp {

struct Vec8 {
    float32x4_t low;
    float32x4_t high;
};

using __m256 = Vec8;

static inline Vec8 _mm256_setzero_ps() {
    return {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
}

static inline Vec8 _mm256_set1_ps(float value) {
    return {vdupq_n_f32(value), vdupq_n_f32(value)};
}

static inline Vec8 _mm256_loadu_ps(const float* values) {
    return {vld1q_f32(values), vld1q_f32(values + 4)};
}

static inline void _mm256_storeu_ps(float* values, Vec8 vector) {
    vst1q_f32(values, vector.low);
    vst1q_f32(values + 4, vector.high);
}

static inline Vec8 _mm256_add_ps(Vec8 first, Vec8 second) {
    return {vaddq_f32(first.low, second.low),
            vaddq_f32(first.high, second.high)};
}

static inline Vec8 _mm256_sub_ps(Vec8 first, Vec8 second) {
    return {vsubq_f32(first.low, second.low),
            vsubq_f32(first.high, second.high)};
}

static inline Vec8 _mm256_mul_ps(Vec8 first, Vec8 second) {
    return {vmulq_f32(first.low, second.low),
            vmulq_f32(first.high, second.high)};
}

static inline Vec8 _mm256_fmadd_ps(Vec8 first, Vec8 second, Vec8 accumulator) {
    return {vfmaq_f32(accumulator.low, first.low, second.low),
            vfmaq_f32(accumulator.high, first.high, second.high)};
}

static inline Vec8 _mm256_fnmadd_ps(Vec8 first, Vec8 second, Vec8 accumulator) {
    return {vfmsq_f32(accumulator.low, first.low, second.low),
            vfmsq_f32(accumulator.high, first.high, second.high)};
}

static inline Vec8 _mm256_min_ps(Vec8 first, Vec8 second) {
    return {vminq_f32(first.low, second.low),
            vminq_f32(first.high, second.high)};
}

static inline Vec8 _mm256_max_ps(Vec8 first, Vec8 second) {
    return {vmaxq_f32(first.low, second.low),
            vmaxq_f32(first.high, second.high)};
}

} // namespace aadsp

#else

#error "The native DSP kernels require AVX2/FMA or AArch64 NEON"

#endif
