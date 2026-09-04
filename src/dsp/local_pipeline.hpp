// Vendored from code/native/shared/local_pipeline.hpp at c26f96b; regenerate with repro/export_rack_config.py.
#pragma once

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "adaa_d7.hpp"
#include "optimized_dsp.hpp"
#include "target_kernels.hpp"

namespace aadsp {

// Clamp: evaluate the pointwise core with the input limited to the target's
// [-clamp, clamp] domain (PolynomialTarget.eval semantics; opt-in, see
// target_kernels.hpp).  Default false keeps the benchmarked lowerings and
// their references unchanged.
template<class Config, bool Clamp = false>
class Local4xPipeline {
public:
    explicit Local4xPipeline(int max_block)
        : max_block_(max_block),
          input_(max_block + input_history_, 0.0f),
          outer_even_(max_block, 0.0f),
          outer_odd_(max_block, 0.0f),
          rate2_(2 * max_block + rate2_history_, 0.0f),
          reconstructed_even_(Config::phase_fused ? 0
                              : 2 * max_block + phase_history_, 0.0f),
          reconstructed_odd_(Config::phase_fused ? 0
                             : 2 * max_block + phase_history_, 0.0f),
          nonlinear_even_(2 * max_block + inner_even_history_, 0.0f),
          nonlinear_odd_(2 * max_block + inner_odd_history_, 0.0f),
          rate2_output_(2 * max_block, 0.0f),
          guard_even_(max_block + guard_even_history_, 0.0f),
          guard_odd_(max_block + guard_odd_history_, 0.0f) {
        if (max_block <= 0) throw std::invalid_argument("max_block must be positive");
        static_assert(Config::core_order >= 0, "core order must be nonnegative");
        static_assert(!Config::phase_fused || Config::core_order == 0,
                      "phase-fused schedule currently supports pointwise cores only");
        static_assert(Config::input_even_size <= max_history_);
        static_assert(Config::input_odd_size <= max_history_);
        static_assert(Config::midpoint_size <= max_history_);
        static_assert(Config::inner_size <= max_history_);
        static_assert(Config::guard_size <= max_history_);
        static_assert(Config::midpoint_delay <= max_history_);
        static_assert(Config::inner_shift <= max_history_);
        static_assert(Config::guard_shift <= max_history_);
        static_assert(Config::midpoint_filtered_phase == 0
                      || Config::midpoint_filtered_phase == 1);
        static_assert(Config::inner_fir_phase == 0 || Config::inner_fir_phase == 1);
        static_assert(Config::guard_fir_phase == 0 || Config::guard_fir_phase == 1);
    }

    void reset() {
        std::fill(input_.begin(), input_.end(), 0.0f);
        std::fill(rate2_.begin(), rate2_.end(), 0.0f);
        std::fill(reconstructed_even_.begin(), reconstructed_even_.end(), 0.0f);
        std::fill(reconstructed_odd_.begin(), reconstructed_odd_.end(), 0.0f);
        std::fill(nonlinear_even_.begin(), nonlinear_even_.end(), 0.0f);
        std::fill(nonlinear_odd_.begin(), nonlinear_odd_.end(), 0.0f);
        std::fill(guard_even_.begin(), guard_even_.end(), 0.0f);
        std::fill(guard_odd_.begin(), guard_odd_.end(), 0.0f);
    }

    void process_block(const float* input, float* output, int samples) {
        if (samples < 0 || samples > max_block_)
            throw std::invalid_argument("block exceeds configured maximum");
        if (samples == 0) return;

        float* input_current = input_.data() + input_history_;
        std::copy(input, input + samples, input_current);
        if constexpr (Config::input_paired) {
            firsym_pair_block(outer_even_.data(), outer_odd_.data(),
                              input_current,
                              Config::input_even, Config::input_even_size,
                              Config::input_odd, Config::input_odd_size,
                              samples);
        } else {
            firsym_block(outer_even_.data(), input_current,
                         Config::input_even, Config::input_even_size, samples);
            firsym_block(outer_odd_.data(), input_current,
                         Config::input_odd, Config::input_odd_size, samples);
        }

        float* rate2_current = rate2_.data() + rate2_history_;
        for (int n = 0; n < samples; ++n) {
            rate2_current[2 * n] = outer_even_[n];
            rate2_current[2 * n + 1] = outer_odd_[n];
        }
        commit_history(input_, samples, input_history_);

        float* nonlinear_even = nonlinear_even_.data() + inner_even_history_;
        float* nonlinear_odd = nonlinear_odd_.data() + inner_odd_history_;
        const int rate2_samples = 2 * samples;

        if constexpr (Config::phase_fused) {
            using Target = typename Config::target;
            if constexpr (Config::midpoint_filtered_phase == 0) {
                firsym_poly_block<Target, Clamp>(nonlinear_even, rate2_current,
                                          Config::midpoint,
                                          Config::midpoint_size,
                                          rate2_samples);
                poly_block<Target, Clamp>(nonlinear_odd,
                                   rate2_current - Config::midpoint_delay,
                                   rate2_samples);
            } else {
                poly_block<Target, Clamp>(nonlinear_even,
                                   rate2_current - Config::midpoint_delay,
                                   rate2_samples);
                firsym_poly_block<Target, Clamp>(nonlinear_odd, rate2_current,
                                          Config::midpoint,
                                          Config::midpoint_size,
                                          rate2_samples);
            }
        } else {
            float* recon_even = reconstructed_even_.data() + phase_history_;
            float* recon_odd = reconstructed_odd_.data() + phase_history_;
            float* direct = Config::midpoint_filtered_phase == 0
                ? recon_odd : recon_even;
            float* filtered = Config::midpoint_filtered_phase == 0
                ? recon_even : recon_odd;
            for (int n = 0; n < rate2_samples; ++n)
                direct[n] = rate2_current[n - Config::midpoint_delay];
            firsym_block(filtered, rate2_current, Config::midpoint,
                         Config::midpoint_size, rate2_samples);
            apply_materialized_core(nonlinear_even, nonlinear_odd, rate2_samples);
        }
        commit_history(rate2_, rate2_samples, rate2_history_);

        const float* inner_fir = Config::inner_fir_phase == 0
            ? nonlinear_even : nonlinear_odd;
        const float* inner_direct = Config::inner_fir_phase == 0
            ? nonlinear_odd : nonlinear_even;
        firsym_add_delayed_block(rate2_output_.data(), inner_fir, inner_direct,
                                 Config::inner, Config::inner_size,
                                 Config::inner_direct_gain, Config::inner_shift,
                                 rate2_samples);
        commit_history(nonlinear_even_, rate2_samples, inner_even_history_);
        commit_history(nonlinear_odd_, rate2_samples, inner_odd_history_);

        float* guard_even = guard_even_.data() + guard_even_history_;
        float* guard_odd = guard_odd_.data() + guard_odd_history_;
        for (int n = 0; n < samples; ++n) {
            guard_even[n] = rate2_output_[2 * n];
            guard_odd[n] = rate2_output_[2 * n + 1];
        }

        const float* guard_fir = Config::guard_fir_phase == 0
            ? guard_even : guard_odd;
        const float* guard_direct = Config::guard_fir_phase == 0
            ? guard_odd : guard_even;
        firsym_add_delayed_block(output, guard_fir, guard_direct,
                                 Config::guard, Config::guard_size,
                                 Config::guard_direct_gain, Config::guard_shift,
                                 samples);
        commit_history(guard_even_, samples, guard_even_history_);
        commit_history(guard_odd_, samples, guard_odd_history_);
    }

private:
    static constexpr int max_history_ = 512;
    static constexpr int input_history_ =
        std::max(Config::input_even_size, Config::input_odd_size) - 1;
    static constexpr int rate2_history_ =
        std::max(Config::midpoint_size - 1, Config::midpoint_delay);
    // History carried by each reconstructed phase buffer for the phase-pair
    // ADAA kernels (deepest lag P reaches (P+1)/2 samples into a phase).
    static constexpr int phase_history_ =
        Config::core_order == 0 ? 0 : (Config::core_order + 1) / 2;
    static constexpr int inner_even_history_ = Config::inner_fir_phase == 0
        ? Config::inner_size - 1 : Config::inner_shift;
    static constexpr int inner_odd_history_ = Config::inner_fir_phase == 1
        ? Config::inner_size - 1 : Config::inner_shift;
    static constexpr int guard_even_history_ = Config::guard_fir_phase == 0
        ? Config::guard_size - 1 : Config::guard_shift;
    static constexpr int guard_odd_history_ = Config::guard_fir_phase == 1
        ? Config::guard_size - 1 : Config::guard_shift;

    static void commit_history(std::vector<float>& storage, int samples,
                               int history) {
        if (history == 0) return;
        float* current = storage.data() + history;
        if (samples >= history) {
            std::memcpy(storage.data(), current + samples - history,
                        history * sizeof(float));
            return;
        }
        std::memmove(storage.data(), storage.data() + samples,
                     (history - samples) * sizeof(float));
        std::memcpy(storage.data() + history - samples, current,
                    samples * sizeof(float));
    }

    void apply_materialized_core(float* nonlinear_even, float* nonlinear_odd,
                                 int rate2_samples) {
        const float* recon_even = reconstructed_even_.data() + phase_history_;
        const float* recon_odd = reconstructed_odd_.data() + phase_history_;
        if constexpr (Config::core_order == 0) {
            poly_block<typename Config::target, Clamp>(nonlinear_even, recon_even,
                                                rate2_samples);
            poly_block<typename Config::target, Clamp>(nonlinear_odd, recon_odd,
                                                rate2_samples);
        } else {
            // Phase-pair ADAA evaluates the interleaved rate-4 operator
            // directly on the two phase buffers; no interleave/deinterleave
            // round trip is materialized.
            apply_adaa_phases(nonlinear_even, nonlinear_odd,
                              recon_even, recon_odd, rate2_samples);
            commit_history(reconstructed_even_, rate2_samples, phase_history_);
            commit_history(reconstructed_odd_, rate2_samples, phase_history_);
        }
    }

    static void apply_adaa_phases(float* nonlinear_even, float* nonlinear_odd,
                                  const float* recon_even,
                                  const float* recon_odd, int samples) {
        using Target = typename Config::target;
        constexpr int order = Config::core_order == 0 ? 1 : Config::core_order;
        const float* phases[2] = {recon_even, recon_odd};
        adaa_poly_rphase_block<Target, order, 2, 0>(nonlinear_even, phases,
                                                    samples);
        adaa_poly_rphase_block<Target, order, 2, 1>(nonlinear_odd, phases,
                                                    samples);
    }

    int max_block_;
    std::vector<float> input_;
    std::vector<float> outer_even_;
    std::vector<float> outer_odd_;
    std::vector<float> rate2_;
    std::vector<float> reconstructed_even_;
    std::vector<float> reconstructed_odd_;
    std::vector<float> nonlinear_even_;
    std::vector<float> nonlinear_odd_;
    std::vector<float> rate2_output_;
    std::vector<float> guard_even_;
    std::vector<float> guard_odd_;
};

} // namespace aadsp
