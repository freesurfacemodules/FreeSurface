// KronVerb: N=16 complex-valued lossless FDN with a fast Kronecker feedback
// operator, per-line phasor (SSB frequency shift) layer, and unimodular
// in-loop nonlinearities.
//
// Integrates three systems:
//  - PhiReverb's complex unitary signal path (L=Re, R=Im stereo, eigenphase control)
//  - Coppola (DAFx26): recursive Kronecker 2x2 kernels, O(N log N) feedback,
//    audio-rate angle modulation, even/odd partition freeze, cross-coupling
//  - Dal Santo et al. (DAFx26): in-loop shimmer operations, re-derived in the
//    complex domain where possible so they are exactly energy-preserving:
//      ring mod        -> uniform phasor ramp   = SSB frequency shift (unimodular)
//      CFWR waveshaper -> Kerr self-phase mod   y = x * e^(i*gamma*|x|) (unimodular)
//      time compression-> dual-tap ring-buffer transposer on odd lines (eq-power xfade)
//
// The whole parameter surface is continuous: no topology rebuilds, no mode
// switches. Freeze is a continuous ramp of (input gain, absorption, damping
// bypass, innermost kernel angle) on the even-indexed lines.

#include "plugin.hpp"

namespace kronverb {

static const int N = 16;          // network size (2^M)
static const int M = 4;           // log2(N) Kronecker levels
static const int BUF = 32768;     // delay buffer length (power of 2)
static const int BUF_MASK = BUF - 1;
static const int WIN = 2048;      // shimmer grain window (samples)
static const float MAX_DELAY = (float)(BUF - WIN - 8);
static const float PHI = 1.61803398875f;
static const float DELAY_EXP = 0.4f;  // m_i = m0 * PHI^(DELAY_EXP * i)

// ---------------------------------------------------------------------------
struct Cplx {
    float re = 0.f;
    float im = 0.f;
};

static inline Cplx cadd(Cplx a, Cplx b) { return {a.re + b.re, a.im + b.im}; }
static inline Cplx cmul(Cplx a, Cplx b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline Cplx cscale(Cplx a, float s) { return {a.re * s, a.im * s}; }

// ---------------------------------------------------------------------------
// Complex circular delay buffer with integer and cubic-interpolated reads.
struct DelayLine {
    Cplx buf[BUF] = {};
    int w = 0;

    void push(Cplx x) {
        buf[w] = x;
        w = (w + 1) & BUF_MASK;
    }

    Cplx readInt(int d) {
        return buf[(w - d) & BUF_MASK];
    }

    // 4-point Lagrange read at fractional delay d (d >= 2). Used for the
    // always-moving shimmer taps, where interpolation loss is masked by
    // crossfading and absorption.
    Cplx readCubic(float d) {
        int i1 = (int)d;
        float t = d - (float)i1;
        Cplx y0 = readInt(i1 - 1);
        Cplx y1 = readInt(i1);
        Cplx y2 = readInt(i1 + 1);
        Cplx y3 = readInt(i1 + 2);
        // Lagrange coefficients for taps at t offsets {-1, 0, 1, 2}
        float c0 = -t * (t - 1.f) * (t - 2.f) / 6.f;
        float c1 = (t + 1.f) * (t - 1.f) * (t - 2.f) / 2.f;
        float c2 = -(t + 1.f) * t * (t - 2.f) / 2.f;
        float c3 = (t + 1.f) * t * (t - 1.f) / 6.f;
        Cplx y;
        y.re = c0 * y0.re + c1 * y1.re + c2 * y2.re + c3 * y3.re;
        y.im = c0 * y0.im + c1 * y1.im + c2 * y2.im + c3 * y3.im;
        return y;
    }
};

// ---------------------------------------------------------------------------
// First-order Thiran allpass for the static main tap: exactly unit-magnitude,
// so a frozen partition (absorption = 1) is truly lossless.
// Fractional delay is kept in [0.5, 1.5] around the integer part.
struct ThiranFrac {
    float C = 0.f;
    Cplx px, py;

    void setFrac(float d) {
        C = (d - 1.f) / (d + 1.f);
    }

    Cplx process(Cplx x) {
        Cplx y;
        y.re = C * (py.re - x.re) + px.re;
        y.im = C * (py.im - x.im) + px.im;
        px = x;
        py = y;
        return y;
    }
};

// ---------------------------------------------------------------------------
// One-pole lowpass on a complex signal (feedback damping).
struct CplxOnePole {
    float a = 0.f;
    Cplx y;

    Cplx process(Cplx x) {
        y.re = (1.f - a) * x.re + a * y.re;
        y.im = (1.f - a) * x.im + a * y.im;
        return y;
    }
};

// ---------------------------------------------------------------------------
// Fast Kronecker mixer: Psi = K_M (x) ... (x) K_1 applied in place in
// O(N log N) (Coppola Alg. 2). Kernel entries are stored complex so the
// structure is ready for full U(2) kernels (expander); the macro layer
// currently drives real rotations.
struct KronMixer {
    struct Kernel {
        Cplx a, b, c, d;
    };
    Kernel K[M];

    // Level l kernel as a plane rotation by theta with an optional kernel
    // phase (kernel = e^(i*phase) * Rot(theta)); phase = 0 keeps it real.
    void setRotation(int level, float theta, float phase = 0.f) {
        float ct = std::cos(theta);
        float st = std::sin(theta);
        float cp = std::cos(phase);
        float sp = std::sin(phase);
        Kernel& k = K[level];
        k.a = {ct * cp, ct * sp};
        k.b = {-st * cp, -st * sp};
        k.c = {st * cp, st * sp};
        k.d = {ct * cp, ct * sp};
    }

    // In-place butterfly. Level l pairs indices differing in bit l, so level 0
    // is the only level mixing even<->odd lines (the freeze partition).
    void apply(Cplx v[N]) {
        for (int l = 0; l < M; l++) {
            const Kernel& k = K[l];
            int h = 1 << l;
            for (int base = 0; base < N; base += 2 * h) {
                for (int j = 0; j < h; j++) {
                    int i0 = base + j;
                    int i1 = base + j + h;
                    Cplx x0 = v[i0];
                    Cplx x1 = v[i1];
                    v[i0] = cadd(cmul(k.a, x0), cmul(k.b, x1));
                    v[i1] = cadd(cmul(k.c, x0), cmul(k.d, x1));
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Dual-read-head shimmer transposer branch: two sawtooth taps with a
// synchronized equal-power crossfade (sin^2 + cos^2 = 1), a DC blocker, and
// pyFDN-style slow energy normalization (two power envelope followers and a
// smoothed, capped sqrt(Pin/Pout) gain; cf. pyFDN td/operators.py DCBlocker
// correct_loss).  The normalization is referenced to the static main tap the
// branch blends against, making the branch power-neutral inside the loop:
// time-compressed reads and correlated tap sums can no longer compound per
// pass, so recirculation cannot run away for any shimmer setting.
struct ShimmerTap {
    float phase = 0.f;
    Cplx dcX, dcY;
    float inPow = 1e-12f, outPow = 1e-12f, comp = 1.f;

    void resetState() {
        inPow = outPow = 1e-12f;
        comp = 1.f;
        dcX = dcY = {0.f, 0.f};
    }

    // rate = (ratio - 1) / WIN per sample; ref is the static main tap whose
    // power is the normalization target; aEnv/aGain are the per-sample
    // envelope coefficients (50 ms / 20 ms time constants).
    Cplx process(DelayLine& line, float baseDelay, float rate, float aEnv,
                 float aGain, Cplx ref) {
        phase -= rate;
        phase -= std::floor(phase);
        float pB = phase + 0.5f;
        pB -= std::floor(pB);
        float gA = std::sin((float)M_PI * phase);
        float gB = std::sin((float)M_PI * pB);
        Cplx sA = line.readCubic(baseDelay + (float)WIN * phase);
        Cplx sB = line.readCubic(baseDelay + (float)WIN * pB);
        Cplx sh = cadd(cscale(sA, gA), cscale(sB, gB));

        // DC blocker y[n] = x[n] - x[n-1] + R y[n-1] on both components:
        // recirculated grains can pile up near-DC content in the loop.
        const float R = 0.995f;
        Cplx y = {sh.re - dcX.re + R * dcY.re, sh.im - dcX.im + R * dcY.im};
        dcX = sh;
        dcY = y;

        const float eps = 1e-12f;
        inPow = aEnv * inPow + (1.f - aEnv) * (ref.re * ref.re + ref.im * ref.im);
        outPow = aEnv * outPow + (1.f - aEnv) * (y.re * y.re + y.im * y.im);
        // Attenuate-only (cap 1): a boosting cap turns the normalizer into a
        // junk amplifier when the shifted content is annihilated -- with
        // negative pitch the signal marches into the DC blocker within a few
        // passes, outPow collapses, and a x4 cap then amplifies crossfade
        // residue every pass (4 * gamma * eps > 1 self-oscillates).  In-loop,
        // referenced to the main tap, the normalizer's job is purely
        // protective: branch power <= reference power.
        float target = std::sqrt((inPow + eps) / (outPow + eps));
        if (target > 1.f) target = 1.f;
        comp = aGain * comp + (1.f - aGain) * target;
        return cscale(y, comp);
    }
};

// ---------------------------------------------------------------------------
// One-pole parameter smoother.
struct Smoother {
    float y = 0.f;
    float k = 0.002f;

    void setTau(float tauSec, float fs) {
        k = 1.f - std::exp(-1.f / (tauSec * fs));
    }

    float process(float target) {
        y += k * (target - y);
        return y;
    }
};

} // namespace kronverb

using namespace kronverb;

// ---------------------------------------------------------------------------
struct KronVerbModule : Module {
    enum ParamIds {
        PARAM_SIZE,
        PARAM_DECAY,
        PARAM_DAMP,
        PARAM_DIFFUSE,
        PARAM_SHIFT,
        PARAM_WARP,
        PARAM_SHIMMER,
        PARAM_PITCH,
        PARAM_FREEZE,
        PARAM_WIDTH,
        PARAM_MIX,
        PARAM_DRIFT,
        PARAM_MIRROR,
        NUM_PARAMS
    };
    enum InputIds {
        INPUT_L,
        INPUT_R,
        CV_SIZE,
        CV_DECAY,
        CV_DIFFUSE,
        CV_SHIFT,
        CV_WARP,
        CV_SHIMMER,
        CV_FREEZE,
        CV_PITCH,
        CV_MIRROR,
        NUM_INPUTS
    };
    enum OutputIds {
        OUTPUT_L,
        OUTPUT_R,
        NUM_OUTPUTS
    };
    enum LightIds {
        NUM_LIGHTS
    };

    DelayLine lines[N];
    ThiranFrac thiran[N];
    CplxOnePole damp[N];
    KronMixer mixer;

    float delaySamp[N] = {};   // current (slewed) delay per line, in samples
    int delayInt[N] = {};      // integer part handed to the buffer read

    ShimmerTap shim[N];        // shimmer branch state (odd lines used)

    // Phasor layer state: per-line phase accumulators whose rates toggle
    // under MIRROR (phase-continuous frequency switching).
    float phTheta[N] = {};
    float mirRate[N];          // smoothed per-line rate multiplier
    int mirState[N];           // toggle bit per line
    int mirTimer[N];           // samples until the next flip
    uint32_t mirRng = 0x9e3779b9u;
    float driftPhase1 = 0.f;
    float driftPhase2 = 0.f;

    Smoother smSize, smFreeze, smDiffuse, smShimmer, smWarp, smWidth, smMix, smDrift, smMirror;

    // Complex output weights, unit magnitude, golden-angle phase spread.
    // The tank's lines carry strongly correlated content (structured mixing),
    // and under MIRROR the relative phase between the SSB and mirrored halves
    // sweeps slowly: a real +-1 output pattern lets the summed output drift
    // through anti-phase and fade by 30+ dB.  Diverse fixed output phases make
    // the sum behave incoherently (diffuse-field-like), holding the envelope
    // within a few dB at any shift rate (verified in sim_mirror2).
    Cplx cOut[N];

    KronVerbModule() {
        config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
        configParam(PARAM_SIZE, 0.f, 1.f, 0.5f, "Size");
        configParam(PARAM_DECAY, 0.f, 1.f, 0.5f, "Decay");
        configParam(PARAM_DAMP, 0.f, 1.f, 0.7f, "Damping");
        configParam(PARAM_DIFFUSE, 0.f, 1.f, 1.f, "Diffusion");
        configParam(PARAM_SHIFT, -1.f, 1.f, 0.f, "Frequency shift");
        configParam(PARAM_WARP, 0.f, 1.f, 0.f, "Warp (Kerr)");
        configParam(PARAM_SHIMMER, 0.f, 1.f, 0.f, "Shimmer");
        configParam(PARAM_PITCH, -12.f, 12.f, 12.f, "Shimmer pitch", " st");
        configParam(PARAM_FREEZE, 0.f, 1.f, 0.f, "Freeze");
        configParam(PARAM_WIDTH, 0.f, 1.f, 1.f, "Width");
        configParam(PARAM_MIX, 0.f, 1.f, 0.35f, "Mix");
        configParam(PARAM_DRIFT, 0.f, 1.f, 0.f, "Drift");
        configParam(PARAM_MIRROR, 0.f, 1.f, 0.f, "Mirror (sideband spread)");

        configInput(INPUT_L, "Left audio");
        configInput(INPUT_R, "Right audio");
        configInput(CV_SIZE, "Size CV");
        configInput(CV_DECAY, "Decay CV");
        configInput(CV_DIFFUSE, "Diffusion CV");
        configInput(CV_SHIFT, "Frequency shift CV");
        configInput(CV_WARP, "Warp CV");
        configInput(CV_SHIMMER, "Shimmer CV");
        configInput(CV_FREEZE, "Freeze CV");
        configInput(CV_PITCH, "Shimmer pitch CV");
        configInput(CV_MIRROR, "Mirror CV");
        configOutput(OUTPUT_L, "Left audio");
        configOutput(OUTPUT_R, "Right audio");
        configBypass(INPUT_L, OUTPUT_L);
        configBypass(INPUT_R, OUTPUT_R);

        for (int i = 0; i < N; i++) {
            float chi = 2.f * (float)M_PI * std::fmod(0.61803398875f * (float)i, 1.f);
            cOut[i] = {std::cos(chi), std::sin(chi)};
            mirRate[i] = 1.f;
            mirState[i] = i & 1;             // start half the lines mirrored
            mirTimer[i] = 977 * (i + 1);     // staggered first flips
        }

        onSampleRateChange();
    }

    void onSampleRateChange() override {
        float fs = APP->engine->getSampleRate();
        smSize.setTau(0.05f, fs);
        smFreeze.setTau(0.02f, fs);
        smDiffuse.setTau(0.01f, fs);
        smShimmer.setTau(0.01f, fs);
        smWarp.setTau(0.01f, fs);
        smWidth.setTau(0.01f, fs);
        smMix.setTau(0.01f, fs);
        smDrift.setTau(0.05f, fs);
        smMirror.setTau(0.01f, fs);
    }

    float macro(int paramId, int cvId, float lo = 0.f, float hi = 1.f) {
        float v = params[paramId].getValue();
        if (inputs[cvId].isConnected())
            v += inputs[cvId].getVoltage() * (hi - lo) * 0.1f;
        return clamp(v, lo, hi);
    }

    void process(const ProcessArgs& args) override {
        float fs = args.sampleRate;

        // -- Macro layer -----------------------------------------------------
        float size = smSize.process(macro(PARAM_SIZE, CV_SIZE));
        float decay = macro(PARAM_DECAY, CV_DECAY);
        float dampV = params[PARAM_DAMP].getValue();
        float diffuse = smDiffuse.process(macro(PARAM_DIFFUSE, CV_DIFFUSE));
        float shiftV = macro(PARAM_SHIFT, CV_SHIFT, -1.f, 1.f);
        float warp = smWarp.process(macro(PARAM_WARP, CV_WARP));
        float shimmer = smShimmer.process(macro(PARAM_SHIMMER, CV_SHIMMER));
        float pitch = macro(PARAM_PITCH, CV_PITCH, -12.f, 12.f);
        float freeze = smFreeze.process(macro(PARAM_FREEZE, CV_FREEZE));
        float width = smWidth.process(params[PARAM_WIDTH].getValue());
        float mix = smMix.process(params[PARAM_MIX].getValue());
        float drift = smDrift.process(params[PARAM_DRIFT].getValue());
        float mirror = smMirror.process(macro(PARAM_MIRROR, CV_MIRROR));

        float t60 = 0.1f * std::pow(600.f, decay);          // 0.1 s .. 60 s
        float dampFc = 200.f * std::pow(100.f, dampV);      // 200 Hz .. 20 kHz
        float shiftHz = shiftV * shiftV * shiftV * 500.f;   // cubic taper, +/-500 Hz
        float ratio = std::pow(2.f, pitch / 12.f);
        float kerrK = warp * 1.2f;                          // rad per volt of |x|

        // -- Delay lengths: phi-geometric series, scaled by SIZE -------------
        // Base 1.5 ms .. 18 ms; spread PHI^(0.4*15) ~ 17.9x.
        float m0 = 0.0015f * std::pow(12.f, size) * fs;
        for (int i = 0; i < N; i++) {
            float m = m0 * std::pow(PHI, DELAY_EXP * (float)i);
            m = clamp(m, 8.f, MAX_DELAY);
            delaySamp[i] = m;
            int di = (int)std::floor(m + 0.5f) - 1;  // frac in [0.5, 1.5]
            if (di < 2) di = 2;
            delayInt[i] = di;
            thiran[i].setFrac(m - (float)di);
        }

        // -- Feedback matrix angles ------------------------------------------
        // DIFFUSE sweeps all levels 0 (parallel combs) -> pi/4 (Hadamard-like).
        // FREEZE scales level 0 toward zero to decouple the even/odd partition.
        // DRIFT wobbles the middle levels to break fixed resonances.
        driftPhase1 += 2.f * (float)M_PI * 0.11f / fs;
        driftPhase2 += 2.f * (float)M_PI * 0.23f / fs;
        if (driftPhase1 > 2.f * (float)M_PI) driftPhase1 -= 2.f * (float)M_PI;
        if (driftPhase2 > 2.f * (float)M_PI) driftPhase2 -= 2.f * (float)M_PI;
        float thetaBase = diffuse * (float)M_PI / 4.f;
        float driftAmt = drift * 0.12f * (float)M_PI;
        mixer.setRotation(0, thetaBase * (1.f - freeze));
        mixer.setRotation(1, thetaBase + driftAmt * std::sin(driftPhase1));
        mixer.setRotation(2, thetaBase + driftAmt * std::sin(driftPhase2));
        mixer.setRotation(3, thetaBase);

        // -- Phasor layer: per-line rates with MIRROR as frequency toggling --
        // Each line has its own phase accumulator theta_i; z_i = e^{i theta_i}
        // is exactly unimodular for every setting, so the phasor layer is
        // lossless at any MIRROR/DIFFUSE combination (a coherent DSB
        // multiplier z = cos + i*lambda*sin has |z| <= 1 and its ~ -6 dB/pass
        // average loss killed the tail once diffusion mixed every packet
        // through the mirrored lines; its make-up gain is parametrically
        // unstable in-loop -- see the core tests).  MIRROR instead toggles
        // each line's rate between +shift and (1 - 2*mirror)*shift at
        // staggered random intervals (~100-300 ms), phase-continuously (an
        // FSK glide, no clicks).  Recirculating packets accumulate a random
        // +-shift step per pass, giving the same binomial spectral diffusion
        // as true DSB -- mirrored sidebands appear across the line ensemble
        // from the first pass -- without tremolo, interference collapse, or
        // energy loss.  Being unimodular it is safe on the frozen partition:
        // a frozen tank keeps diffusing spectrally, losslessly.
        float kRate = 1.f - std::exp(-1.f / (fs * 0.01f));  // 10 ms rate slew
        float phInc = 2.f * (float)M_PI * shiftHz / fs;

        // -- Damping coefficient ---------------------------------------------
        float dampA = std::exp(-2.f * (float)M_PI * dampFc / fs);
        for (int i = 0; i < N; i++)
            damp[i].a = dampA;

        // -- Read taps + per-line loop operations ----------------------------
        float shimRate = (ratio - 1.f) / (float)WIN;
        float aEnv = std::exp(-1.f / (fs * 0.05f));   // 50 ms power envelopes
        float aGain = std::exp(-1.f / (fs * 0.02f));  // 20 ms gain smoothing
        Cplx v[N];
        for (int i = 0; i < N; i++) {
            // Main tap: integer delay + Thiran allpass (lossless).
            Cplx tap = thiran[i].process(lines[i].readInt(delayInt[i]));

            // Energy-normalized shimmer transposer branch on odd lines, mixed
            // as an equal-power PARALLEL send: the static tap always stays at
            // full pre-normalization amplitude, so the reverb bed persists at
            // any SHIMMER setting (a crossfade replaced the static tap
            // entirely at 1.0, consuming the unshifted reverb within a few
            // recirculations).  At full knob the odd lines split 50/50, i.e.
            // a quarter of the network's recirculating energy is shifted per
            // pass -- matching the density of pyFDN's 2-of-8 longest-channel
            // configuration -- and each pass binomially splits energy across
            // the octave ladder: the classic recursive shimmer cascade.
            if (i & 1) {
                if (shimmer > 1e-4f) {
                    Cplx sh = shim[i].process(lines[i], delaySamp[i], shimRate,
                                              aEnv, aGain, tap);
                    float norm = 1.f / std::sqrt(1.f + shimmer * shimmer);
                    tap.re = (tap.re + shimmer * sh.re) * norm;
                    tap.im = (tap.im + shimmer * sh.im) * norm;
                } else {
                    shim[i].resetState();
                }
            }

            // Kerr self-phase modulation: y = x * e^(i*k*|x|). |y| = |x|
            // exactly, so this waveshaper cannot add or remove energy.
            if (kerrK > 1e-5f) {
                float mag = std::sqrt(tap.re * tap.re + tap.im * tap.im);
                float ph = kerrK * mag;
                tap = cmul(tap, {std::cos(ph), std::sin(ph)});
            }

            // Absorption (delay-proportional T60) and damping filter.
            // Even lines ramp to lossless/bypassed as freeze rises.
            float g = std::exp(-6.90776f * delaySamp[i] / (t60 * fs));
            Cplx damped = damp[i].process(tap);
            if (!(i & 1)) {
                g += freeze * (1.f - g);
                damped.re = damped.re + freeze * (tap.re - damped.re);
                damped.im = damped.im + freeze * (tap.im - damped.im);
            }
            if (--mirTimer[i] <= 0) {
                mirRng ^= mirRng << 13;
                mirRng ^= mirRng >> 17;
                mirRng ^= mirRng << 5;
                mirState[i] ^= 1;
                mirTimer[i] = (int)(fs * (0.1f + 0.2f * (float)(mirRng >> 8) / 16777216.f));
            }
            float rTarget = 1.f - 2.f * mirror * (float)mirState[i];
            mirRate[i] += kRate * (rTarget - mirRate[i]);
            phTheta[i] += phInc * mirRate[i];
            if (phTheta[i] > (float)M_PI) phTheta[i] -= 2.f * (float)M_PI;
            if (phTheta[i] < -(float)M_PI) phTheta[i] += 2.f * (float)M_PI;
            Cplx zi = {std::cos(phTheta[i]), std::sin(phTheta[i])};
            v[i] = cscale(cmul(damped, zi), g);
        }

        // -- Unitary mixing ---------------------------------------------------
        mixer.apply(v);

        // -- Input / output ---------------------------------------------------
        float inL = inputs[INPUT_L].getVoltage();
        float inR = inputs[INPUT_R].getVoltage();
        Cplx x = {inL, inR};
        float bGain = 0.25f;  // 1/sqrt(N)

        Cplx wet = {0.f, 0.f};
        for (int i = 0; i < N; i++) {
            wet = cadd(wet, cmul(v[i], cOut[i]));

            float ing = bGain;
            if (!(i & 1))
                ing *= (1.f - freeze);  // frozen partition takes no input
            lines[i].push(cadd(v[i], cscale(x, ing)));
        }
        wet = cscale(wet, 0.5f);  // 2/sqrt(N) makeup

        // Width: scale the Im (side-like) component of the complex output.
        float wetL = wet.re;
        float wetR = wet.re + width * (wet.im - wet.re);

        float dryG = std::cos(mix * (float)M_PI / 2.f);
        float wetG = std::sin(mix * (float)M_PI / 2.f);
        outputs[OUTPUT_L].setVoltage(dryG * inL + wetG * wetL);
        outputs[OUTPUT_R].setVoltage(dryG * inR + wetG * wetR);
    }
};

// ---------------------------------------------------------------------------
struct KronVerbWidget : ModuleWidget {
    KronVerbWidget(KronVerbModule* module) {
        setModule(module);
        setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Causality.svg")));

        const float col1 = 18.f, col2 = 53.f, col3 = 88.f;
        const float y0 = 30.f, dy = 30.f;

        // Column 1: audio I/O + CV
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 0 * dy), module, KronVerbModule::INPUT_L));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 1 * dy), module, KronVerbModule::INPUT_R));
        addOutput(createOutputCentered<PJ301MPort>(Vec(col1, y0 + 2 * dy), module, KronVerbModule::OUTPUT_L));
        addOutput(createOutputCentered<PJ301MPort>(Vec(col1, y0 + 3 * dy), module, KronVerbModule::OUTPUT_R));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 4 * dy), module, KronVerbModule::CV_SIZE));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 5 * dy), module, KronVerbModule::CV_DECAY));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 6 * dy), module, KronVerbModule::CV_DIFFUSE));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 7 * dy), module, KronVerbModule::CV_SHIFT));

        // Column 2: core macros
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 0 * dy), module, KronVerbModule::PARAM_SIZE));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 1 * dy), module, KronVerbModule::PARAM_DECAY));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 2 * dy), module, KronVerbModule::PARAM_DAMP));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 3 * dy), module, KronVerbModule::PARAM_DIFFUSE));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 4 * dy), module, KronVerbModule::PARAM_SHIFT));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 5 * dy), module, KronVerbModule::PARAM_WARP));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 6 * dy), module, KronVerbModule::PARAM_SHIMMER));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 7 * dy), module, KronVerbModule::PARAM_PITCH));

        // Column 3: remaining macros + CV
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col3, y0 + 0 * dy), module, KronVerbModule::PARAM_FREEZE));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col3, y0 + 1 * dy), module, KronVerbModule::PARAM_WIDTH));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col3, y0 + 2 * dy), module, KronVerbModule::PARAM_MIX));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col3, y0 + 3 * dy), module, KronVerbModule::PARAM_DRIFT));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(col2, y0 + 8 * dy), module, KronVerbModule::PARAM_MIRROR));
        addInput(createInputCentered<PJ301MPort>(Vec(col1, y0 + 8 * dy), module, KronVerbModule::CV_MIRROR));
        addInput(createInputCentered<PJ301MPort>(Vec(col3, y0 + 4 * dy), module, KronVerbModule::CV_WARP));
        addInput(createInputCentered<PJ301MPort>(Vec(col3, y0 + 5 * dy), module, KronVerbModule::CV_SHIMMER));
        addInput(createInputCentered<PJ301MPort>(Vec(col3, y0 + 6 * dy), module, KronVerbModule::CV_FREEZE));
        addInput(createInputCentered<PJ301MPort>(Vec(col3, y0 + 7 * dy), module, KronVerbModule::CV_PITCH));
    }
};

Model* modelKronVerb = createModel<KronVerbModule, KronVerbWidget>("FreeSurface-KronVerb");
