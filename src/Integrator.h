#ifndef FREESURFACE_INTEGRATOR_H
#define FREESURFACE_INTEGRATOR_H

#include "WaterTable2Constants.h"
#include "rack.hpp"
#include <cmath>
#include <array>

using namespace rack;
using simd::float_4;
using simd::int32_4;

// Variation on smoothstep with the max value of the first derivative always <= 1.0
// This clamps without also amplifying the signal.
static inline float_4 smoothclamp(float_4 x, float_4 low, float_4 high) {
    x = (2.f/3.f) * x;
    x = simd::clamp((x - low) / (high - low), 0.f, 1.f);
    return simd::rescale(x*x*(3.f - 2.f*x),0.f,1.f,low,high);
}

// Variation on smoothstep with the max value of the first derivative always <= 1.0
// This clamps without also amplifying the signal.
// TODO: it's nice that this is pretty fast, but I think a nicer-sounding alternative is feasible
static inline float smoothclamp(float x, float low, float high) {
    x = (2.f/3.f) * x;
    x = simd::clamp((x - low) / (high - low), 0.f, 1.f);
    return simd::rescale(x*x*(3.f - 2.f*x),0.f,1.f,low,high);
}

// TODO: make sure to multiply feedback param by 4 for squid axon
// TODO: should I just put everything in ModelIter_Data?
// TODO: make sure to control safe_damping with the safe_timestep function below
template <size_t channel_size>
struct alignas(16) ModelParams {
    std::array<float_4, channel_size> kernel_weight_N;
    std::array<float_4, channel_size> kernel_weight_E;
    std::array<float_4, channel_size> kernel_weight_W;
    std::array<float_4, channel_size> kernel_weight_S;
    std::array<float_4, channel_size> kernel_weight_C;
    std::array<float_4, channel_size> input_probe_L_window;
    std::array<float_4, channel_size> input_probe_R_window;
    std::array<float_4, channel_size> output_probe_L_window;
    std::array<float_4, channel_size> output_probe_R_window;

    float amp_in_L;
    float amp_in_R;
    float low_cut;
    float amp_in_prev_L;
    float amp_in_prev_R;
    float timestep;
    float safe_damping;
    float feedback;
    float damping;
    float decay;
    float pos_in_R;
    float sig_in_R;
    float amp_out_L;
    float amp_out_R;

    /*
     * I should rethink the upsampling/downsampling process. The biquad approach works OK,
     * but needs a bit too low of a cutoff.  A combo of lagrange interpolation plus a 1 pole
     * filter might actually work better for upsampling. For downsampling I should look at
     * other IIR filters.
     * I'd prefer not to include the filters in this struct, but they're used in the inner
     * solver loop to control feedback. Should evaluate if that's needed.
     */
    dsp::BiquadFilter biquad_output_L;
    dsp::BiquadFilter biquad_output_R;
    dsp::BiquadFilter biquad_input_L;
    dsp::BiquadFilter biquad_input_R;

    ModelParams() {
        const float biquad_cutoff = 0.0625f;
        const float biquad_Q = 0.5f;
        const float biquad_gain = 1.0f;
        biquad_input_L.setParameters(dsp::BiquadFilter::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);
        biquad_input_R.setParameters(dsp::BiquadFilter::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);
        biquad_output_L.setParameters(dsp::BiquadFilter::Type::LOWPASS, biquad_cutoff, biquad_Q, biquad_gain);
        biquad_output_R.setParameters(dsp::BiquadFilter::Type::LOWPASS, biquad_cutoff, biquad_Q, biquad_gain);
    }
};

template<size_t N, size_t channel_size>
struct alignas(16) ModelIter_Data {
    std::array<std::array<float_4, channel_size>, N> inputs;
    std::array<std::array<float_4, channel_size>, N> outputs;

    std::array<std::array<float_4, channel_size>, N> half_1;
    std::array<std::array<float_4, channel_size>, N> half_2;
    std::array<std::array<float_4, channel_size>, N> half_3;
    std::array<std::array<float_4, channel_size>, N> half_4;

    std::array<std::array<float_4, channel_size>, N> grad_1;
    std::array<std::array<float_4, channel_size>, N> grad_2;
    std::array<std::array<float_4, channel_size>, N> grad_3;
    std::array<std::array<float_4, channel_size>, N> grad_4;

    std::array<std::array<float_4, channel_size>, N> t_laplacian;
    std::array<std::array<float_4, channel_size>, N> t_gradient_x;
    std::array<std::array<float_4, channel_size>, N> t_gradient_y;
    std::array<std::array<float_4, channel_size>, N> t_deriv_3_x;
    std::array<std::array<float_4, channel_size>, N> t_deriv_3_y;

    std::array<std::array<float_4, channel_size>, N> v_dc;

    // These will point to entire 2D arrays; treat them as single pointers
    std::array<std::array<float_4, channel_size>, N>* init;
    std::array<std::array<float_4, channel_size>, N>* grad;

    // Scalar members
    float input_L = 0.f;
    float input_R = 0.f;
    float t_amp_out_L = 0.f;
    float t_amp_out_R = 0.f;
    unsigned int iter = 0;
    bool ping = false;

    void setPing() {
        ping = !ping;
    }

    std::array<std::array<float_4, channel_size>, N>& getInput() {
        if (ping) {
            return inputs;
        } else {
            return outputs;
        }
    }

    std::array<std::array<float_4, channel_size>, N>& getOutput() {
        if (!ping) {
            return inputs;
        } else {
            return outputs;
        }
    }

    // **Constructor**
    ModelIter_Data() {
        // Initialize arrays with zero-initialized float_4
        for (size_t i = 0; i < N; ++i) {
            inputs[i].fill(float_4(0.f));
            outputs[i].fill(float_4(0.f));

            half_1[i].fill(float_4(0.f));
            half_2[i].fill(float_4(0.f));
            half_3[i].fill(float_4(0.f));
            half_4[i].fill(float_4(0.f));

            grad_1[i].fill(float_4(0.f));
            grad_2[i].fill(float_4(0.f));
            grad_3[i].fill(float_4(0.f));
            grad_4[i].fill(float_4(0.f));

            t_laplacian[i].fill(float_4(0.f));
            t_gradient_x[i].fill(float_4(0.f));
            t_gradient_y[i].fill(float_4(0.f));
            t_deriv_3_x[i].fill(float_4(0.f));
            t_deriv_3_y[i].fill(float_4(0.f));

            v_dc[i].fill(float_4(0.f));
        }

        // Initialize pointers to nullptr or appropriate arrays
        init = nullptr; // Will be assigned later
        grad = nullptr; // Will be assigned later
    }
};

template<size_t N, size_t channel_size>
using ModelPointer = void (*)(ModelIter_Data<N, channel_size>&, ModelParams<channel_size>&);

#define I_CLAMP 30.f
#define F_CLAMP 30.f
#define INTER_CLAMP(x) smoothclamp((x),-clamp_range[j],clamp_range[j])
#define FINAL_CLAMP(x) smoothclamp((x),-clamp_range[j],clamp_range[j])
#define FB_CLAMP(x) smoothclamp((x), -I_CLAMP, I_CLAMP)

static inline float sum(float_4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

template <size_t N, size_t channel_size>
struct WaveEquationFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {

        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 a = (*data.init)[0][i];
            float_4 b = (*data.init)[1][i];

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            float_4 summed_probe_input = probe_in_L + probe_in_R;

            (*data.grad)[0][i] = (summed_probe_input + b +
                                  params.safe_damping * params.damping * data.t_laplacian[0][i] - params.decay * a -
                                  data.v_dc[0][i]);
            (*data.grad)[1][i] = (data.t_laplacian[0][i] - params.decay * b - data.v_dc[1][i]);

            data.v_dc[0][i] = simd::crossfade(a, data.v_dc[0][i], 0.9995);
            data.v_dc[1][i] = simd::crossfade(b, data.v_dc[1][i], 0.9995);

        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

template <size_t N, size_t channel_size>
struct SquidAxonFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {

        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        // Squid axon params
        float k1 = 1.0 - params.decay;
        const float k2 = 0.0;
        const float k3 = 1.0;
        const float k4 = 1.0;
        const float epsilon = 0.1;
        const float ak0 = -0.1;
        const float ak1 = 2.0;

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 a = simd::clamp((*data.init)[0][i], -2.0f, 2.0f);
            float_4 b = simd::clamp((*data.init)[1][i], -2.0f, 2.0f);

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            float_4 summed_probe_input = probe_in_L + probe_in_R;

            (*data.grad)[0][i] = summed_probe_input + k1 * a - k2 * a * a - k4 * a * a * a - b +
                                 params.safe_damping * data.t_laplacian[0][i];
            (*data.grad)[1][i] = -summed_probe_input + epsilon * (k3 * a - ak1 * b - ak0) +
                                 params.safe_damping * params.damping * data.t_laplacian[1][i];
        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

template <size_t N, size_t channel_size>
struct SchrodingerFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {
        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 a = (*data.init)[0][i];
            float_4 b = (*data.init)[1][i];

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            float_4 summed_probe_input = probe_in_L + probe_in_R;

            // Schrodinger equation, with added diffusion and decay
            (*data.grad)[0][i] = (-summed_probe_input - data.t_laplacian[1][i] - params.decay * a + params.safe_damping * params.damping * data.t_laplacian[0][i]);
            (*data.grad)[1][i] = ( summed_probe_input + data.t_laplacian[0][i] - params.decay * b + params.safe_damping * params.damping * data.t_laplacian[1][i]);
        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

template <size_t N, size_t channel_size>
struct ShallowWaterFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {
        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);


        const float g = 9.8;
        //const float k = 0.03;

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 a = (*data.init)[0][i];
            float_4 b = (*data.init)[1][i];
            float_4 c = (*data.init)[2][i];

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            //float_4 summed_probe_input = probe_in_L + probe_in_R;

            /*
                delta_a = - max(0, a - d) * (x_gradient_b + y_gradient_c);
                delta_b = - g * x_gradient_a - k * b;
                delta_c = - g * y_gradient_a - k * c;
             */

            // Shallow water equation, with added diffusion and decay
            // Do I need to take the max here? Does it make more sense to use probe as depth value or A value?
            /*(*data.grad)[0][i] = summed_probe_input - fmax(0.f, a - summed_probe_input) * (data.t_gradient_x[1][i] + data.t_gradient_y[2][i])
                    + params.safe_damping * params.damping * data.t_laplacian[0][i]
                    - params.decay * a;*/
            (*data.grad)[0][i] = - fmax(0.f,a + 1.f) * (data.t_gradient_x[1][i] + data.t_gradient_y[2][i])
                                 + params.safe_damping * params.damping * data.t_laplacian[0][i]
                                 - params.decay * a;
            (*data.grad)[1][i] = probe_in_L - g * data.t_gradient_x[0][i] - params.decay * b;
            (*data.grad)[2][i] = probe_in_R - g * data.t_gradient_y[0][i] - params.decay * c;

        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

template <size_t N, size_t channel_size>
struct YangJumpingFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {
        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        const float D_a = 1.0f;
        const float D_b = 1.0f;
        const float D_c = 60.0f;
        const float k1  = -8.5f;
        const float k3  = 10.0f;
        const float k4  = 2.0f;
        const float tau = 50.0f;

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 a = (*data.init)[0][i];
            float_4 b = (*data.init)[1][i];
            float_4 c = (*data.init)[2][i];

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            float_4 summed_probe_input = probe_in_L + probe_in_R;

            /*
                delta_a = D_a * laplacian_a + k1 + 2.0f*a - a*a*a - k3*b - k4*c;
                delta_b = D_b * laplacian_b + ( a - b ) / tau;
                delta_c = D_c * laplacian_c + a - c;
             */

            // Yang jumping oscillons
            (*data.grad)[0][i] = simd::clamp(summed_probe_input + D_a * params.safe_damping * params.damping * data.t_laplacian[0][i]
                    + k1 + 2.0f*a - a*a*a - k3*b - k4*c, -10.f, 10.f);
            (*data.grad)[1][i] = simd::clamp(D_b * params.safe_damping * params.damping * data.t_laplacian[1][i]
                    + (a - b) / tau, -10.f, 10.f);
            (*data.grad)[2][i] = simd::clamp(D_c * params.safe_damping * params.damping * data.t_laplacian[2][i]
                    + a - c, -10.f, 10.f);

        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

template <size_t N, size_t channel_size>
struct EulerCompressibleFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {
        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        const float ts = 0.4f;

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];
            float_4 probe_in_R = data.input_R * params.input_probe_R_window[i];

            float_4 vx = (*data.init)[0][i];
            float_4 vy = (*data.init)[1][i];
            float_4 p  = (*data.init)[2][i];

            probe_out_L += p * params.output_probe_L_window[i];
            probe_out_R += p * params.output_probe_R_window[i];

            //float_4 summed_probe_input = probe_in_L + probe_in_R;

            // Compressible euler equations
            (*data.grad)[0][i] = - ts * (vx * data.t_gradient_x[0][i] + vy * data.t_gradient_y[0][i]) // -u.grad(u)
                                 - ts * data.t_gradient_x[2][i] // -grad(p)   (may be missing a constant here)
                                 + params.safe_damping * params.damping * data.t_laplacian[0][i]
                                 + probe_in_L;
            (*data.grad)[1][i] = - ts * (vx * data.t_gradient_x[1][i] + vy * data.t_gradient_y[1][i]) // -u.grad(u)
                                 - ts * data.t_gradient_y[2][i] // -grad(p)   (may be missing a constant here)
                                 + params.safe_damping * params.damping * data.t_laplacian[1][i]
                                 + probe_in_R;
            (*data.grad)[2][i] = - ts * (vx * data.t_gradient_x[2][i] + vy * data.t_gradient_y[2][i]) // - u.grad(p)
                                 - ts * (data.t_gradient_x[0][i] + data.t_gradient_y[1][i]) // - p*div(u)
                                 + params.safe_damping * params.damping * data.t_laplacian[2][i] // extra diffusion term
                                 - params.decay * p;

        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

// TODO: will need x and y gradients
template <size_t N, size_t channel_size>
struct AdvectionFunctor {
    void operator()(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params) const {

        float_4 probe_out_L = float_4(0.0);
        float_4 probe_out_R = float_4(0.0);

        float_4 probe_in_R = simd::rescale(data.input_R, -10.0, 10.0, -1.0, 1.0);
        float pos_to_mod_offset = 2.0 * (params.pos_in_R - (MAX_POSITION / 2.0)) / MAX_POSITION;
        probe_in_R = params.sig_in_R * (probe_in_R + pos_to_mod_offset);

        for (unsigned int i = 0; i < channel_size; i++) {
            float_4 probe_in_L = data.input_L * params.input_probe_L_window[i];

            float_4 a = (*data.init)[0][i];

            probe_out_L += a * params.output_probe_L_window[i];
            probe_out_R += a * params.output_probe_R_window[i];

            // additive mode necessarily works differently here
            float_4 summed_probe_input = probe_in_L;

            // gradients are smoothclamped for stability
            // TODO: evaluate whether a better gradient clipping scheme might sound better
            (*data.grad)[0][i] = smoothclamp(
                    (summed_probe_input + params.safe_damping * params.damping * data.t_laplacian[0][i] -
                     params.decay * a)  // input
                    - probe_in_R * data.t_gradient_x[0][i], -10.0f, 10.0f); // advection

        }

        data.t_amp_out_L = sum(probe_out_L);
        data.t_amp_out_R = sum(probe_out_R);
    }
};

/** We need to compute the gradient and laplacian of two float_4
 * 	buffers several times here, so we need to do it as fast as possible.
 * 	To do this, each float_4 buffer is circular shifted by one float to
 *  the left and one float to the right to get the left and right
 *  neighbors needed for the calculation aligned into float_4s.
 *  We use hardware shuffle intrinsics to swizzle _m128 float vectors
 *  to get left-shifted and right-shifted vectors. After this point,
 *  we can do our simple calculations using SIMD ops.
*/
/*
    TODO: let's do templating here so we can create
     function specializations that only compute the derivatives needed.
     This should open up the possibility for more models as well.
    TODO: if we make the laplacian kernel weights modifiable, the structure
     of the kernel weight data has big performance implications.
     At the moment the most sensible thing seems to be to have separate arrays
     of float_4s for N,S,E,W cardinal directions
    TODO: performance doesn't improve here when using LTO,
     or vary significantly depending on the mode, which suggests the compiler isn't
     able to discover which derivatives go unused here. I really don't want to do it this way,
     but it may be necessary to implement derivative calculations in the model functors themselves
     to mitigate this. If that's what's needed, this should probably be split into
     separate functions for each derivative, and those functions should get the result for
     a single float_4 at a time, inside the inner loop of the model functors. Ideally, we
     want all the reads to be shared, but this is challenging to structure if the functions are split up,
     and we most of all don't want to allocate any heap memory to do so, nor hint the compiler
     that the loop must be executed serially.
 */
#define INDEX_MASK_X static_cast<unsigned int>(CHANNEL_MASK_X)
#define INDEX_MASK_Y static_cast<unsigned int>(CHANNEL_MASK_Y)
// can these be simplified?
#define INDEX_Y_BASE (index / CHANNEL_SIZE_X)
#define INDEX_X_BASE (index & INDEX_MASK_X)
#define INDEX_X_MINUS_1 (INDEX_Y_BASE * CHANNEL_SIZE_X + ((index-1) & INDEX_MASK_X))
#define INDEX_X_PLUS_1 (INDEX_Y_BASE * CHANNEL_SIZE_X + ((index+1) & INDEX_MASK_X))
#define INDEX_Y_MINUS_1 (((INDEX_Y_BASE-1) & INDEX_MASK_Y) * CHANNEL_SIZE_X + INDEX_X_BASE)
#define INDEX_Y_PLUS_1 (((INDEX_Y_BASE+1) & INDEX_MASK_Y) * CHANNEL_SIZE_X + INDEX_X_BASE)
#define INDEX_Y_MINUS_2 (((INDEX_Y_BASE-2) & INDEX_MASK_Y) * CHANNEL_SIZE_X + INDEX_X_BASE)
#define INDEX_Y_PLUS_2 (((INDEX_Y_BASE+2) & INDEX_MASK_Y) * CHANNEL_SIZE_X + INDEX_X_BASE)

#ifdef __APPLE__
    typedef float v4sf __attribute__((__vector_size__(16)));
    typedef int v4si __attribute__((__vector_size__(16)));
#else
    typedef float v4sf __attribute__ ((vector_size (16)));
    typedef int v4si __attribute__ ((vector_size (16)));
    const v4si mask_l = {1,2,3,4};
    const v4si mask_r = {3,4,5,6};
    const v4si mask_l_2 = {2,3,4,5};
    const v4si mask_r_2 = {2,3,4,5};
#endif
// TODO: these conversions probably compile to noop, but I'm not 100% sure if that's the case for the float_4 constructor
#define V4SF_TO_FLOAT_4(v) float_4(reinterpret_cast<__m128>(v))
#define FLOAT_4_TO_V4SF(f) reinterpret_cast<v4sf>(f.v)
template <size_t channel_size>
static inline void gradient_and_laplacian(
        const std::array<float_4, channel_size> &x,
        std::array<float_4, channel_size> &grad_out_x,
        std::array<float_4, channel_size> &grad_out_y,
        std::array<float_4, channel_size> &grad3_out_x,
        std::array<float_4, channel_size> &grad3_out_y,
        std::array<float_4, channel_size> &lapl_out,
        const ModelParams<channel_size>& params
    ) {
    for (unsigned int index = 0; index < channel_size; index++) {
        v4sf e = FLOAT_4_TO_V4SF(x[INDEX_X_PLUS_1]);
        v4sf w = FLOAT_4_TO_V4SF(x[INDEX_X_MINUS_1]);
        v4sf c = FLOAT_4_TO_V4SF(x[index]);
        float_4 n =  x[INDEX_Y_PLUS_1];
        float_4 s =  x[INDEX_Y_MINUS_1];
        float_4 n2 = x[INDEX_Y_PLUS_2];
        float_4 s2 = x[INDEX_Y_MINUS_2];

        // shuffle left means align the float4 so we get the east value
        // shuffle right means align the float4 so we get the west value
#ifdef __APPLE__
        float_4 shuffle_l = V4SF_TO_FLOAT_4(__builtin_shufflevector(c, e, 1, 2, 3, 4));
        float_4 shuffle_r = V4SF_TO_FLOAT_4(__builtin_shufflevector(w, c, 3, 4, 5, 6));
        float_4 shuffle_l_2 = V4SF_TO_FLOAT_4(__builtin_shufflevector(c, e, 2, 3, 4, 5));
        float_4 shuffle_r_2 = V4SF_TO_FLOAT_4(__builtin_shufflevector(w, c, 2, 3, 4, 5));
#else
        float_4 shuffle_l = V4SF_TO_FLOAT_4(__builtin_shuffle(c, e, mask_l));
        float_4 shuffle_r = V4SF_TO_FLOAT_4(__builtin_shuffle(w, c, mask_r));
        float_4 shuffle_l_2 = V4SF_TO_FLOAT_4(__builtin_shuffle(c, e, mask_l_2));
        float_4 shuffle_r_2 = V4SF_TO_FLOAT_4(__builtin_shuffle(w, c, mask_r_2));
#endif
        // TODO: how to handle weights for odd-numbered derivatives?
        grad_out_x[index] = 0.5f * (shuffle_l - shuffle_r);
        grad_out_y[index] = 0.5f * (n - s);

        grad3_out_x[index] = - 0.5f * shuffle_r_2 + shuffle_r - shuffle_l + 0.5f * shuffle_l_2;
        grad3_out_y[index] = - 0.5f * s2 + s - n + 0.5f * n2;

        lapl_out[index] = params.kernel_weight_N[index] * n
                + params.kernel_weight_S[index] * s
                + params.kernel_weight_E[index] * shuffle_l
                + params.kernel_weight_W[index] * shuffle_r
                + params.kernel_weight_C[index] * x[index];
    }
}

template <size_t channel_size>
static inline void processInputSample(float &input_L, float &input_R, const float &feedback_amp_L, const float &feedback_amp_R, int iter, ModelParams<channel_size>& params) {
    input_L = params.feedback * FB_CLAMP(0.25 * feedback_amp_L) + params.biquad_input_L.process(input_L);
    input_R = params.feedback * FB_CLAMP(0.25 * feedback_amp_R) + params.biquad_input_R.process(input_R);
}

template <size_t channel_size>
static inline void processOutputSample(float &sample_L, float &sample_R, int iter, ModelParams<channel_size>& params) {
    sample_L = params.biquad_output_L.process(sample_L);
    sample_R = params.biquad_output_R.process(sample_R);
}

template <size_t N, size_t channel_size, typename ModelFunctor>
static inline void modelIteration(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params, ModelFunctor model) {

    // TODO: it probably makes sense to just put all of my data in a single templated struct instead of this
    processInputSample<channel_size>(data.input_L, data.input_R, data.t_amp_out_L, data.t_amp_out_R, data.iter, params);

    // TODO: Does -O3 always unroll here?
    //  Ideally I'd also specify the needed derivatives with template parameters, but that also might be unnecessary with -O3.
    for (unsigned int i = 0; i < N; i++) {
        gradient_and_laplacian<channel_size>((*data.init)[i], data.t_gradient_x[i], data.t_gradient_y[i], data.t_deriv_3_x[i], data.t_deriv_3_y[i], data.t_laplacian[i], params);
    }

    model(data, params);

    processOutputSample(data.t_amp_out_L, data.t_amp_out_R, data.iter, params);
}

/** Runge-Kutta (RK4) integration,
 *  Using the 3/8s Runge Kutta method.
 *  Used to increase stability at large timesteps
 *  and also to upsample our input.
 *  TODO: if I figure out how to claw back some performance elsewhere, consider
 *   using a higher-order method here. Implicit solvers are probably off the table though
 *  TODO: consider instead using two multidimensional vectors (for ping ponging)
 *   so we can expand the number of parameters more freely
 */
template <size_t N, size_t channel_size, typename ModelFunctor>
static inline void RK4_iter_3_8s(ModelIter_Data<N, channel_size>& data, ModelParams<channel_size>& params, ModelFunctor model, std::array<float, N> clamp_range) {

    const float third = 1.0/3.0;

    data.init = &data.getInput(); // returns a reference
    data.grad = &data.grad_1;
    data.input_L = params.amp_in_L - params.low_cut * params.amp_in_prev_L;
    data.input_R = params.amp_in_R - params.low_cut * params.amp_in_prev_R;
    data.iter = 0;

    modelIteration<N, channel_size, ModelFunctor>(data, params, model);

    // TODO: in the original version, the clamping was not actually applied in these stages due to a bug. Maybe I can actually do without.
    for (unsigned int i = 0; i < channel_size; i++) {
        for (unsigned int j = 0; j < N; j++) {
            data.half_2[j][i] = INTER_CLAMP(data.getInput()[j][i] + third * params.timestep * data.grad_1[j][i]);
            //data.half_2[j][i] = tanhLut.get(data.getInput()[j][i] + third * params.timestep * data.grad_1[j][i]);
        }
    }

    data.init = &data.half_2;
    data.grad = &data.grad_2;
    data.input_L = 0.;
    data.input_R = 0.;
    data.iter = 1;

    // Round 2, 1/3 step
    // input is only non-zero on the first round for upsampling
    // TODO: let's do lagrange interpolation instead
    modelIteration<N, channel_size, ModelFunctor>(data, params, model);

    for (unsigned int i = 0; i < channel_size; i++) {
        for (unsigned int j = 0; j < N; j++) {
            data.half_3[j][i] = INTER_CLAMP(data.getInput()[j][i] + params.timestep * (-third * data.grad_1[j][i] + data.grad_2[j][i]));
            //data.half_3[j][i] = tanhLut.get(data.getInput()[j][i] + params.timestep * (-third * data.grad_1[j][i] + data.grad_2[j][i]));
        }
    }

    data.init = &data.half_3;
    data.grad = &data.grad_3;
    data.input_L = 0.;
    data.input_R = 0.;
    data.iter = 2;

    // Round 3, 2/3 step
    modelIteration<N, channel_size, ModelFunctor>(data, params, model);


    for (unsigned int i = 0; i < channel_size; i++) {
        for (unsigned int j = 0; j < N; j++) {
            data.half_4[j][i] = INTER_CLAMP(data.getInput()[j][i] + params.timestep * (data.grad_1[j][i] - data.grad_2[j][i] + data.grad_3[j][i]));
            //data.half_4[j][i] = tanhLut.get(data.getInput()[j][i] + params.timestep * (data.grad_1[j][i] - data.grad_2[j][i] + data.grad_3[j][i]));
        }
    }

    data.init = &data.half_4;
    data.grad = &data.grad_4;
    data.input_L = 0.;
    data.input_R = 0.;
    data.iter = 3;

    // Round 4, whole step
    modelIteration<N, channel_size, ModelFunctor>(data, params, model);


    for (unsigned int i = 0; i < channel_size; i++) {
        for (unsigned int j = 0; j < N; j++) {
            data.getOutput()[j][i] = FINAL_CLAMP(data.getInput()[j][i] + params.timestep * (data.grad_1[j][i] + 3.f * data.grad_2[j][i] + 3.f * data.grad_3[j][i] + data.grad_4[j][i]) / 8.f);
            //data.getOutput()[j][i] = tanhLut.get(data.getInput()[j][i] + params.timestep * (data.grad_1[j][i] + 3.f * data.grad_2[j][i] + 3.f * data.grad_3[j][i] + data.grad_4[j][i]) / 8.f);
        }
    }

    params.amp_out_L = math::clamp(data.t_amp_out_L,-100.0f,100.0f);
    params.amp_out_R = math::clamp(data.t_amp_out_R,-100.0f,100.0f);

}


#endif //FREESURFACE_INTEGRATOR_H
