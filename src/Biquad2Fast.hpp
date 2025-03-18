#pragma once
#include <cmath>

/**
 * @brief Various filter modes used as a template parameter.
 */
enum class BiquadMode {
    Lowpass,
    Highpass,
    Bandpass,
    Notch
};

/**
 * @brief A fast Biquad filter with template-based mode selection.
 * Usage:
 *   FastBiquadFilter<BiquadMode::Lowpass> myFilter;
 *   myFilter.freqTarget      = 0.1;         // normalized freq
 *   myFilter.resonanceTarget = 2.0;         // for example
 *   myFilter.recalcCoefficients();         // once, or whenever parameters update
 *   myFilter.processBlock(...);            // per audio block
 *
 */
template <BiquadMode Mode>
struct FastBiquadFilter
{
    //======================================================================
    // Internal state for the biquad (per channel)
    //======================================================================
    double z1L = 0.0, z2L = 0.0, z3L = 0.0, z4L = 0.0;  ///< Left channel delay samples
    double z1R = 0.0, z2R = 0.0, z3R = 0.0, z4R = 0.0;  ///< Right channel delay samples

    //======================================================================
    // Biquad coefficients for the *current* freq/res
    //======================================================================
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    double b1 = 0.0, b2 = 0.0;

    //======================================================================
    // Parameter chasing (smoothed parameters)
    //======================================================================
    double freqChase       = 0.0015;
    double resonanceChase  = 1.0;

    //======================================================================
    // Target parameters (set externally, updated rarely)
    //======================================================================
    double freqTarget       = 0.0015; // Normalized frequency [0..0.5], or your chosen range
    double resonanceTarget  = 1.0;    // Q factor or damping, typically >= 1.0

    double sampleRate       = 44100.0; // If freqTarget is in Hz, you'd do freqTarget = desiredHz / (sampleRate*2).

    // Chasing speeds for each parameter
    double freqChaseSpeed       = 1000.0;
    double resonanceChaseSpeed  = 1000.0;
    double outputChaseSpeed     = 1000.0;
    double wetChaseSpeed        = 1000.0;

    //======================================================================
    // Recompute the filter coefficients for the target freq/res
    //======================================================================
    void recalcCoefficients()
    {
        // Compute K and norm for the current freqTarget/resonanceTarget
        // (If freqTarget is in [0..0.5], do K=tan(M_PI*freqTarget). If in Hz, do freqTarget/= (sampleRate*2)).
        double K = std::tan(M_PI * freqChase);
        double norm = 1.0 / (1.0 + (K / resonanceChase) + (K*K));

        if constexpr (Mode == BiquadMode::Lowpass) {
            a0 = (K*K) * norm;
            a1 = 2.0 * a0;
            a2 = a0;
            b1 = 2.0 * ((K*K) - 1.0) * norm;
            b2 = (1.0 - (K/resonanceChase) + (K*K)) * norm;
        }
        else if constexpr (Mode == BiquadMode::Highpass) {
            a0 = norm;
            a1 = -2.0 * norm;
            a2 = norm;
            b1 = 2.0 * ((K*K) - 1.0) * norm;
            b2 = (1.0 - (K/resonanceChase) + (K*K)) * norm;
        }
        else if constexpr (Mode == BiquadMode::Bandpass) {
            a0 = (K / resonanceChase) * norm;
            a1 = 0.0;
            a2 = -a0;
            b1 = 2.0 * ((K*K) - 1.0) * norm;
            b2 = (1.0 - (K / resonanceChase) + (K*K)) * norm;
        }
        else { // BiquadMode::Notch
            a0 = (1.0 + (K*K)) * norm;
            a1 = 2.0 * ((K*K) - 1.0) * norm;
            a2 = a0;
            b1 = a1;
            b2 = (1.0 - (K / resonanceChase) + (K*K)) * norm;
        }
    }

    //======================================================================
    // Process a block of stereo samples in-place or to outLeft/outRight
    //======================================================================
    void processBlock(const float* inLeft,
                      const float* inRight,
                      float*       outLeft,
                      float*       outRight,
                      int          numSamples)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            // Read input
            double inputL = (double)inLeft[i];
            double inputR = (double)inRight[i];

            // Parameter chasing
            freqChase       += (freqTarget       - freqChase)       * (1.0/freqChaseSpeed);
            resonanceChase  += (resonanceTarget  - resonanceChase)  * (1.0/resonanceChaseSpeed);
            
            recalcCoefficients();
            
            double outL = (a0 * inputL) + (a1 * z1L) + (a2 * z2L) - (b1 * z3L) - (b2 * z4L);
            z2L = z1L;
            z1L = inputL;
            z4L = z3L;
            z3L = outL;

            double outR = (a0 * inputR) + (a1 * z1R) + (a2 * z2R) - (b1 * z3R) - (b2 * z4R);
            z2R = z1R;
            z1R = inputR;
            z4R = z3R;
            z3R = outR;

            // Write output
            outLeft[i]  = (float)outL;
            outRight[i] = (float)outR;
        }
    }
};
