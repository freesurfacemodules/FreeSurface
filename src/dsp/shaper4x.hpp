#pragma once
// 4x local/guard oversampled waveshaper with a runtime nonlinearity.
//
// The stage structure and alignment are those of the study's verified
// Local4xPipeline (materialized schedule): two-phase FIR interpolation to
// 2x, midpoint reconstruction to 4x, pointwise core, exact-halfband inner
// decimation, guard-aware decimation to 1x.  The core is any callable
// object with apply_block(out, in, n) instead of a compile-time
// polynomial trait, so the module can reshape it from its knobs.  The
// filter coefficients and shifts come from a generated Config
// (generated/lg57_config.hpp).
#include <algorithm>
#include <cstring>
#include <vector>
#include "optimized_dsp.hpp"

namespace polydist {

template<class Config>
class Shaper4x {
public:
    explicit Shaper4x(int max_block)
        : max_block_(max_block),
          input_(max_block + input_history_, 0.0f),
          outer_even_(max_block, 0.0f),
          outer_odd_(max_block, 0.0f),
          rate2_(2 * max_block + rate2_history_, 0.0f),
          recon_even_(2 * max_block, 0.0f),
          recon_odd_(2 * max_block, 0.0f),
          nonlinear_even_(2 * max_block + inner_even_history_, 0.0f),
          nonlinear_odd_(2 * max_block + inner_odd_history_, 0.0f),
          rate2_output_(2 * max_block, 0.0f),
          guard_even_(max_block + guard_even_history_, 0.0f),
          guard_odd_(max_block + guard_odd_history_, 0.0f) {
        static_assert(Config::core_order == 0, "pointwise core only");
    }

    void reset() {
        for (auto* v : {&input_, &rate2_, &nonlinear_even_, &nonlinear_odd_,
                        &guard_even_, &guard_odd_})
            std::fill(v->begin(), v->end(), 0.0f);
    }

    template<class Core>
    void process_block(const float* input, float* output, int samples,
                       const Core& core) {
        if (samples <= 0 || samples > max_block_) return;
        using namespace aadsp;

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

        const int rate2_samples = 2 * samples;
        float* direct = Config::midpoint_filtered_phase == 0
            ? recon_odd_.data() : recon_even_.data();
        float* filtered = Config::midpoint_filtered_phase == 0
            ? recon_even_.data() : recon_odd_.data();
        for (int n = 0; n < rate2_samples; ++n)
            direct[n] = rate2_current[n - Config::midpoint_delay];
        firsym_block(filtered, rate2_current, Config::midpoint,
                     Config::midpoint_size, rate2_samples);
        commit_history(rate2_, rate2_samples, rate2_history_);

        float* nonlinear_even = nonlinear_even_.data() + inner_even_history_;
        float* nonlinear_odd = nonlinear_odd_.data() + inner_odd_history_;
        core.apply_block(nonlinear_even, recon_even_.data(), rate2_samples);
        core.apply_block(nonlinear_odd, recon_odd_.data(), rate2_samples);

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
    static constexpr int input_history_ =
        std::max(Config::input_even_size, Config::input_odd_size) - 1;
    static constexpr int rate2_history_ =
        std::max(Config::midpoint_size - 1, Config::midpoint_delay);
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

    int max_block_;
    std::vector<float> input_, outer_even_, outer_odd_, rate2_;
    std::vector<float> recon_even_, recon_odd_;
    std::vector<float> nonlinear_even_, nonlinear_odd_, rate2_output_;
    std::vector<float> guard_even_, guard_odd_;
};

} // namespace polydist
