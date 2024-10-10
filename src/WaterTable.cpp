#include "plugin.hpp"
#include "OpCache.hpp"
#include "Profiler.hpp"
#include <math.h>
#include <cstring>

using simd::float_4;
using simd::int32_4;

// 2^30 * ln(2)
#define EXP2_30_TO_EXP 744261117.954893018
#define ONE_OVER_SQRT_TWO_PI 0.3989422804
#define SQRT_2 1.41421356237

// must be power-of-two
#define CHANNEL_SIZE 16
#define CHANNEL_SIZE_FLOATS (CHANNEL_SIZE << 2)
#define CHANNEL_MASK (CHANNEL_SIZE - 1)

#define MAX_POSITION (CHANNEL_SIZE * 4.0)


// WARNING! This will generate a HUGE amount of data in the log file 
// when knobs are turned, or CV input is connected to the knob position/sigma
//#define DEBUG_PROBE_PRINT

struct WaveChannel {
	enum Model {
		WAVE_EQUATION,
		SQUID_AXON,
		SCHRODINGER,
		RK4_ADVECTION
	};
	Model model;

	enum ProbeType {
		INTEGRAL,
		DIFFERENTIAL,
		SINC
	};

	enum OversamplingMode {
		OVERSAMPLE_SINC,
		OVERSAMPLE_BIQUAD
	};

	enum ClipRange {
		V_10,
		V_30,
		V_60,
		V_100
	};

	OversamplingMode oversampling_mode = OversamplingMode::OVERSAMPLE_BIQUAD;

	float clip_range = 30.0f;
	ClipRange clip_range_mode = ClipRange::V_30;

	dsp::BiquadFilter biquad_output_L;
	dsp::BiquadFilter biquad_output_R;
	dsp::BiquadFilter biquad_input_L;
	dsp::BiquadFilter biquad_input_R;

	dsp::Decimator<4, 4> decimator_output_L;
	dsp::Decimator<4, 4> decimator_output_R;
	dsp::Upsampler<4, 4> upsampler_input_L;
	dsp::Upsampler<4, 4> upsampler_input_R;

	float upsampler_result_input_L[4] = {0};
	float upsampler_result_input_R[4] = {0};
	float decimator_input_L[4] = {0};
	float decimator_input_R[4] = {0};
        

	/** Member function pointer for the current model.
	 *  Nasty, but using a switch or inheritance would be nastier 
	 *  here and probably worse for performance.
	 */
	typedef void (WaveChannel::*ModelPointer) (
		const std::vector<float_4>&, const std::vector<float_4>&, 
		const std::vector<float_4>&, const std::vector<float_4>&,
		const std::vector<float_4>&, const std::vector<float_4>&,
		std::vector<float_4>&, std::vector<float_4>&, 
		std::vector<float_4>&, std::vector<float_4>&,
		const float&, const float&,
		float&, float&);

	ModelPointer modelPointer;

	float pos_in_L = 0.0;
	float pos_in_R = 0.0;
	float amp_in_L = 0.0; 
	float amp_in_R = 0.0; 
	float sig_in_L = 0.0;
	float sig_in_R = 0.0;
	float pos_out_L = 0.0;
	float pos_out_R = 0.0;
	float sig_out_L = 0.0;
	float sig_out_R = 0.0;
	float amp_out_L = 0.0;
	float amp_out_R = 0.0; 
	float amp_in_prev_L = 0.0;
	float amp_in_prev_R = 0.0;
	float damping = 0.1; 
	float timestep = 0.01;
	float decay = 0.005;
	float feedback = 0.0;
	float low_cut = 0.0;

	// ping pong buffer setup
	bool pong = false;

	ProbeType input_probe_type_L = ProbeType::INTEGRAL;
	ProbeType input_probe_type_R = ProbeType::INTEGRAL;
	ProbeType output_probe_type_L = ProbeType::INTEGRAL;
	ProbeType output_probe_type_R = ProbeType::INTEGRAL;
	bool additive_mode_L = true;
	bool additive_mode_R = true;

	std::vector<float_4> v_a0 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_b0 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_a1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_b1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	//special dc bias cancelling vectors
	std::vector<float_4> v_dc_a = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_dc_b = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

	// temporaries for the model update steps. decreases overhead by declaring them in this scope
	std::vector<float_4> t_gradient_a = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> t_gradient_b = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> t_laplacian_a = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> t_laplacian_b = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

	// weights for the input and output probes, generated whenever the respective probe settings change
	std::vector<float_4> input_probe_L_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> input_probe_R_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> output_probe_L_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> output_probe_R_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

	WaveChannel() {
		model = Model::WAVE_EQUATION;
		modelPointer = &WaveChannel::stepWaveEquation;
		//const float biquad_cutoff = 0.125f;
		const float biquad_cutoff = 0.0625f;
		const float biquad_Q = 0.5f;
		const float biquad_gain = 1.0f;
		biquad_input_L.setParameters(dsp::BiquadFilter::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);
		biquad_input_R.setParameters(dsp::BiquadFilter::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);
		biquad_output_L.setParameters(dsp::BiquadFilter::Type::LOWPASS, biquad_cutoff, biquad_Q, biquad_gain);
		biquad_output_R.setParameters(dsp::BiquadFilter::Type::LOWPASS, biquad_cutoff, biquad_Q, biquad_gain);
	}

	// classic GLSL-style hermite smoothstep function
	inline float_4 smoothstep(float_4 x) {
		x = simd::clamp(x, 0., 1.);
		return x*x*(3. - 2.*x);
	}

	// smoothstep fit to the error function
	// https://www.desmos.com/calculator/molpuljtzy
	inline float_4 smoothstep_erf(float_4 x) {
		const float P = 0.3761264; //fits a smoothstep to error function with equal first derivative at x=0
		x = P*x + 0.5;
		return 2.0 * (smoothstep(x) - 0.5);
	}

	// smoothstep fit to the derivative of the error function (a gaussian)
	// https://www.desmos.com/calculator/molpuljtzy
	inline float_4 smoothstep_erf_deriv(float_4 x) {
		const float P2 = 1.128379172; //fits a smoothstep to error function first derivative with equal second derivative at x=0
		const float P3 = 0.577350268;

		x = P3*x;
		return P2 * (smoothstep(x + 1) - smoothstep(x));
	}

	/** sign function that doesn't return 0.
		we need sign to return something other than 0 for x==0 
		for our signed distance function; the choice of positive 
		or negative is unimportant here. 
		the signed distance function will have an erroneous 
		value at exact integral values of x otherwise. 
	*/
	inline float_4 sgnZ(float_4 x) {
		float_4 signbit = x & -0.f;
		return signbit | 1.f;
	}

	// Finds the closest distance from index to comp in a modular space.
	// INDEX MUST BE IN THE RANGE 0 <= INDEX <= MAX_POSITION
	inline float_4 wrappedSignedDistance(float_4 index, float_4 comp) {
		float_4 c0 = simd::fabs(index - comp);
		float_4 c1 = simd::fabs(index - comp + MAX_POSITION);
		float_4 c2 = simd::fabs(index - comp - MAX_POSITION);
		float_4 abs_min = simd::fmin(c0, simd::fmin(c1, c2));
		float_4 sgn1 = sgnZ(c0 - c1);
		float_4 sgn2 = sgnZ(c0 - c2);
		float_4 sgn3 = sgnZ(c1 - c2);
		return sgn1 * sgn2 * sgn3 * abs_min;
	}

	// Variation on smoothstep with the max value of the first derivative always <= 1.0
	// This clamps without also amplifying the signal.
	inline float_4 smoothclamp(float_4 x, float_4 low, float_4 high) {
		x = (2./3.) * x;
		x = simd::clamp((x - low) / (high - low), 0., 1.);
		return simd::rescale(x*x*(3. - 2.*x),0.,1.,low,high);
	}

	// Variation on smoothstep with the max value of the first derivative always <= 1.0
	// This clamps without also amplifying the signal.
	inline float smoothclamp(float x, float low, float high) {
		x = (2./3.) * x;
		x = simd::clamp((x - low) / (high - low), 0., 1.);
		return simd::rescale(x*x*(3. - 2.*x),0.,1.,low,high);
	}

	/** computes a gaussian using the differences of the (approximate) error function
	 *  this is a much better approach than simply sampling a gaussian because we can
	 *  almost eliminate aliasing, even at small kernel sizes
	 */
	inline float_4 approxGaussian(float_4 mean, float_4 x, float_4 sig) {
		float_4 x_s = wrappedSignedDistance(x, mean);
		float_4 x_d_l = x_s-0.5;
		float_4 x_d_r = x_s+0.5;
		float_4 xsq_l = x_d_l / (SQRT_2*sig);
		float_4 xsq_r = x_d_r / (SQRT_2*sig);
		float_4 erf_l = smoothstep_erf(xsq_l);
		float_4 erf_r = smoothstep_erf(xsq_r);
		
		return 0.5*(erf_r-erf_l);
	}

	/** Same as above, but we get the first derivative of a gaussian using an
	 *  approximation of the first derivative of the error function.
	 */
	inline float_4 approxGaussianDeriv(float_4 mean, float_4 x, float_4 sig) {
		float_4 x_s = wrappedSignedDistance(x, mean);
		float_4 x_d_l = x_s-0.5;
		float_4 x_d_r = x_s+0.5;
		float_4 xsq_l = x_d_l / (SQRT_2*sig);
		float_4 xsq_r = x_d_r / (SQRT_2*sig);
		float_4 erf_l = smoothstep_erf_deriv(xsq_l);
		float_4 erf_r = smoothstep_erf_deriv(xsq_r);
		
		return 0.5*(erf_r-erf_l);
	}

	/** A sinc function.
	 *  Using the integral trick to combat aliasing
	 *  unfortunately doesn't work here.
	 */
	inline float_4 sinc(float_4 center, float_4 x, float_4 sig) {
		float_4 x_s = wrappedSignedDistance(x, center);
		float_4 x_s2 = x_s / sig;
		float_4 snc = simd::sin(x_s2)/x_s2;
		float_4 almost_zero = simd::abs(x_s2) < 1.0e-6f;
		return simd::ifelse(almost_zero, 1.0, snc);
	}

	
	/** We need to compute the gradient and laplacian of two float_4 
	 * 	buffers several times here, so we need to do it as fast as possible.
	 * 	To do this, each float_4 buffer is circular shifted by one float to
	 *  the left and one float to the right to get the left and right
	 *  neighbors needed for the calculation aligned into float_4s.
	 *  We use hardware shuffle intrinsics to swizzle _m128 float vectors
	 *  to get left-shifted and right-shifted vectors. After this point,
	 *  we can do our simple calculations using SIMD ops.
	*/
	#define INDEX_MASK static_cast<unsigned int>(CHANNEL_MASK)
	#define INDEX_MINUS_1 ((index-1) & INDEX_MASK)
	#define INDEX_PLUS_1 ((index+1) & INDEX_MASK)
	#ifdef __APPLE__
		typedef float v4sf __attribute__((__vector_size__(16)));
		typedef int v4si __attribute__((__vector_size__(16)));
		#define V4SF_TO_FLOAT_4(v) float_4(reinterpret_cast<__m128>(v))
		#define FLOAT_4_TO_V4SF(f) reinterpret_cast<v4sf>(f.v)
		inline void gradient_and_laplacian(const std::vector<float_4> &x, std::vector<float_4> &grad_out, std::vector<float_4> &lapl_out) {
			for (int index = 0; index < CHANNEL_SIZE; index++) {
				v4sf e = FLOAT_4_TO_V4SF(x[INDEX_PLUS_1]);
				v4sf w = FLOAT_4_TO_V4SF(x[INDEX_MINUS_1]);
				v4sf c = FLOAT_4_TO_V4SF(x[index]);

				float_4 shuffle_l = V4SF_TO_FLOAT_4(__builtin_shufflevector(c, e, 1, 2, 3, 4));
				float_4 shuffle_r = V4SF_TO_FLOAT_4(__builtin_shufflevector(w, c, 3, 4, 5, 6));

				grad_out[index] = (shuffle_l - shuffle_r) / 2.0;

				lapl_out[index] = shuffle_l + shuffle_r - 2.0 * x[index];
			}
		}
	#else
		typedef float v4sf __attribute__ ((vector_size (16)));
		typedef int v4si __attribute__ ((vector_size (16)));
		#define V4SF_TO_FLOAT_4(v) float_4(reinterpret_cast<__m128>(v))
		#define FLOAT_4_TO_V4SF(f) reinterpret_cast<v4sf>(f.v)
		inline void gradient_and_laplacian(const std::vector<float_4> &x, std::vector<float_4> &grad_out, std::vector<float_4> &lapl_out) {
			v4si mask_l = {1,2,3,4};
			v4si mask_r = {3,4,5,6};
			for (int index = 0; index < CHANNEL_SIZE; index++) {
				v4sf e = FLOAT_4_TO_V4SF(x[INDEX_PLUS_1]);
				v4sf w = FLOAT_4_TO_V4SF(x[INDEX_MINUS_1]);
				v4sf c = FLOAT_4_TO_V4SF(x[index]);

				float_4 shuffle_l = V4SF_TO_FLOAT_4(__builtin_shuffle(c, e, mask_l));
				float_4 shuffle_r = V4SF_TO_FLOAT_4(__builtin_shuffle(w, c, mask_r));

				grad_out[index] = (shuffle_l - shuffle_r) / 2.0;

				lapl_out[index] = shuffle_l + shuffle_r - 2.0 * x[index];
			}
		}
	#endif

	/** Temporaries for RK4 integration. Declaring them in function scope incurs a huge
	 *  overhead cost. Some of these (the _half_ vectors) could be reused, but it's not much
	 *  memory and doing so would make the code significantly more confusing.
	 */
	std::vector<float_4> a_grad_1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_half_2 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_grad_2 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_half_3 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_grad_3 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_half_4 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> a_grad_4 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

	std::vector<float_4> b_grad_1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_half_2 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_grad_2 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_half_3 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_grad_3 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_half_4 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> b_grad_4 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

	#define I_CLAMP clip_range
	#define F_CLAMP clip_range
	#define INTER_CLAMP(x) smoothclamp((x),-I_CLAMP,I_CLAMP)
	#define FINAL_CLAMP(x) smoothclamp((x),-F_CLAMP,F_CLAMP)

	float t_amp_in_prev_L = 0.0f;
	float t_amp_in_prev_R = 0.0f;
	
	
	inline void processInputSample(float &input_L, float &input_R, const float &feedback_amp_L, const float &feedback_amp_R, int iter) {
		float t_amp_in_L, t_amp_in_R;
		if (oversampling_mode == OversamplingMode::OVERSAMPLE_SINC) {
			if (iter == 0) {
				upsampler_input_L.process(input_L, upsampler_result_input_L);
				upsampler_input_R.process(input_R, upsampler_result_input_R);
			}
			t_amp_in_L = upsampler_result_input_L[iter];
			t_amp_in_R = upsampler_result_input_R[iter];
		} else {
			t_amp_in_L = biquad_input_L.process(input_L);
			t_amp_in_R = biquad_input_R.process(input_R);
		}

		// no really special reason for this, but squid axon rapidly squashes
		// inputs, so it can take more feedback
		if (model == WaveChannel::SQUID_AXON) {
			input_L = feedback * 4.0 * INTER_CLAMP(0.25 * feedback_amp_L) + t_amp_in_L;
			input_R = feedback * 4.0 * INTER_CLAMP(0.25 * feedback_amp_R) + t_amp_in_R;
		} else {
			input_L = feedback * INTER_CLAMP(0.25 * feedback_amp_L) + t_amp_in_L;
			input_R = feedback * INTER_CLAMP(0.25 * feedback_amp_R) + t_amp_in_R;
		}

	}

	/*
	inline void processInputSample(float &input_L, float &input_R, const float &feedback_amp_L, const float &feedback_amp_R) {
		input_L = biquad_input_L.process(feedback * INTER_CLAMP(feedback_amp_L) + input_L);
		input_R = biquad_input_R.process(feedback * INTER_CLAMP(feedback_amp_R) + input_R);
	}*/

	inline void processOutputSample(float &sample_L, float &sample_R, int iter) {
		if (oversampling_mode == OversamplingMode::OVERSAMPLE_SINC) {
			decimator_input_L[iter] = sample_L;
			decimator_input_R[iter] = sample_R;
		} else {
			sample_L = biquad_output_L.process(sample_L);
			sample_R = biquad_output_R.process(sample_R);
		}
	}

	inline void modelIteration(
			const std::vector<float_4> &a_in, std::vector<float_4> &delta_a,
			const std::vector<float_4> &b_in, std::vector<float_4> &delta_b,
			float input_L, float input_R, 
			float &t_amp_out_L, float &t_amp_out_R,
			int iter) {

		processInputSample(input_L, input_R, t_amp_out_L, t_amp_out_R, iter);

		gradient_and_laplacian(a_in, t_gradient_a, t_laplacian_a);
		gradient_and_laplacian(b_in, t_gradient_b, t_laplacian_b);
		(this->*modelPointer)(a_in, b_in, t_laplacian_a, t_laplacian_b, 
		t_gradient_a, t_gradient_b, delta_a, delta_b, v_dc_a, v_dc_b,
		input_L, input_R, t_amp_out_L, t_amp_out_R);

		processOutputSample(t_amp_out_L, t_amp_out_R, iter);
	}

	/** Runge-Kutta (RK4) integration,
	 *  Using the 3/8s Runge Kutta method.
	 *  Used to increase stability at large timesteps
	 *  and also to upsample our input.
	 */
	void RK4_iter_3_8s(
			const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
			std::vector<float_4> &a1, std::vector<float_4> &b1) {

		const float third = 1.0/3.0;

		// Round 1, initial step
		modelIteration( a0, a_grad_1, b0, b_grad_1, 
				amp_in_L - low_cut * amp_in_prev_L, 
				amp_in_R - low_cut * amp_in_prev_R, 
				amp_out_L, amp_out_R, 0);

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_2[i] = a0[i] + third * timestep * a_grad_1[i]);
			INTER_CLAMP(b_half_2[i] = b0[i] + third * timestep * b_grad_1[i]);
		}

		// Round 2, 1/3 step
		// input is only non-zero on the first round for upsampling		
		modelIteration( a_half_2, a_grad_2, b_half_2, b_grad_2, 0.f, 0.f, amp_out_L, amp_out_R, 1);

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_3[i] = a0[i] + timestep * (-third * a_grad_1[i] + a_grad_2[i]));
			INTER_CLAMP(b_half_3[i] = b0[i] + timestep * (-third * b_grad_1[i] + b_grad_2[i]));
		}

		// Round 3, 2/3 step
		modelIteration( a_half_3, a_grad_3, b_half_3, b_grad_3, 0.f, 0.f, amp_out_L, amp_out_R, 2);

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_4[i] = a0[i] + timestep * (a_grad_1[i] - a_grad_2[i] + a_grad_3[i]));
			INTER_CLAMP(b_half_4[i] = b0[i] + timestep * (b_grad_1[i] - b_grad_2[i] + b_grad_3[i]));
		}

		// Round 4, whole step
		modelIteration( a_half_4, a_grad_4, b_half_4, b_grad_4, 0.f, 0.f, amp_out_L, amp_out_R, 3);

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			//final result
			//clamping isn't a part of RK4, but it's more convenient to do it here than elsewhere
			a1[i] = FINAL_CLAMP(a0[i] + timestep * (a_grad_1[i] + 3.0f * a_grad_2[i] + 3.0f * a_grad_3[i] + a_grad_4[i]) / 8.0f);
			b1[i] = FINAL_CLAMP(b0[i] + timestep * (b_grad_1[i] + 3.0f * b_grad_2[i] + 3.0f * b_grad_3[i] + b_grad_4[i]) / 8.0f);
		}

		// clamp to prevent blowups, but with a large range to avoid clipping in general
		if (oversampling_mode == OversamplingMode::OVERSAMPLE_SINC) {
			amp_out_L = math::clamp(decimator_output_L.process(decimator_input_L), -100.0f, 100.0f);
			amp_out_R = math::clamp(decimator_output_R.process(decimator_input_R), -100.0f, 100.0f);
		} else {
			amp_out_L = math::clamp(amp_out_L,-100.0f,100.0f);
			amp_out_R = math::clamp(amp_out_R,-100.0f,100.0f);
		}

	}

	float sum(float_4 x) {
		return x[0] + x[1] + x[2] + x[3];
	}

	/* for stability, the coefficient for laplacian components
		should always work out to be <= 0.5. 
		(This limit is complicated somewhat when using RK4, 
		but holds for euler integration and our laplacian stencil. 
		RK4 raises the limit however, so 0.5 is a safe assumption)
	*/
	inline float getSafeTimestep() {
		return std::min(1.0f, 1.0f/(2.0f*timestep));
	}

	void stepWaveEquation(
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		std::vector<float_4> &dc_bias_a, std::vector<float_4> &dc_bias_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
		
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		float safe_timestep = getSafeTimestep();

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			float_4 probe_in_L = t_input_L * input_probe_L_window[i];
			float_4 probe_in_R = t_input_R * input_probe_R_window[i];

			float_4 a = a0[i];
			float_4 b = b0[i];

			probe_out_L += a * output_probe_L_window[i];
			probe_out_R += a * output_probe_R_window[i];
			float_4 summed_probe_input = 
					(additive_mode_L ? probe_in_L : (a * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (a * probe_in_R));

			/*for equality mode below, may be completely implemented in the future
			float_4 summed_probe_input = 
					(additive_mode_L ? probe_in_L : 0.0) + 
					(additive_mode_R ? probe_in_R : 0.0);*/

			delta_a[i] = (summed_probe_input + b + safe_timestep * damping * laplacian_a[i] - decay * a - dc_bias_a[i]);
			delta_b[i] = (laplacian_a[i] - decay * b - dc_bias_b[i]);

			/* experimental "equality" mode
			   this is not the objectively correct way to do this,
			   but that way is much more expensive, either here or
			   on the precomputation side.*/
			/*
			if (!additive_mode_L && !additive_mode_R) {
				delta_a[i] = simd::crossfade(delta_a[i], t_input_L * input_probe_L_window[i] + t_input_R * input_probe_R_window[i], simd::abs(input_probe_L_window[i]) + simd::abs(input_probe_R_window[i]));
			} else if (!additive_mode_R) {
				delta_a[i] = simd::crossfade(delta_a[i], t_input_R * input_probe_R_window[i], simd::abs(input_probe_R_window[i]));
			} else if (!additive_mode_L){
				delta_a[i] = simd::crossfade(delta_a[i], t_input_L * input_probe_L_window[i], simd::abs(input_probe_L_window[i]));
			}*/
			dc_bias_a[i] = simd::crossfade(a, dc_bias_a[i],0.9995);
			dc_bias_b[i] = simd::crossfade(b, dc_bias_b[i],0.9995);

		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void stepSquidAxon(
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		std::vector<float_4> &dc_bias_a, std::vector<float_4> &dc_bias_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {

		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		float safe_timestep = getSafeTimestep();

		// Squid axon params
		float k1 = 1.0-decay;
		const float k2 = 0.0;
		const float k3 = 1.0;
		const float k4 = 1.0;
		const float epsilon = 0.1;
		float delta = safe_timestep * damping;
		const float ak0 = -0.1;
		const float ak1 = 2.0;

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			float_4 probe_in_L = t_input_L * input_probe_L_window[i];
			float_4 probe_in_R = t_input_R * input_probe_R_window[i];

			float_4 a = simd::clamp(a0[i],-2.0f,2.0f);
			float_4 b = simd::clamp(b0[i],-2.0f,2.0f);

			probe_out_L += a * output_probe_L_window[i];
			probe_out_R += a * output_probe_R_window[i];

			float_4 summed_probe_input_a = 
					(additive_mode_L ? probe_in_L : (a * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (a * probe_in_R));
			float_4 summed_probe_input_b = 
					(additive_mode_L ? probe_in_L : (b * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (b * probe_in_R));
			delta_a[i] = summed_probe_input_a + k1*a - k2*a*a - k4*a*a*a - b + safe_timestep * laplacian_a[i];
			delta_b[i] = - summed_probe_input_b + epsilon*(k3*a - ak1*b - ak0) + delta*laplacian_b[i];
		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void stepSchrodinger(
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		std::vector<float_4> &dc_bias_a, std::vector<float_4> &dc_bias_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
				  			
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		float safe_timestep = getSafeTimestep();

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			float_4 probe_in_L = t_input_L * input_probe_L_window[i];
			float_4 probe_in_R = t_input_R * input_probe_R_window[i];

			float_4 a = a0[i];
			float_4 b = b0[i];

			probe_out_L += a * output_probe_L_window[i];
			probe_out_R += a * output_probe_R_window[i];

			float_4 summed_probe_input_a = 
					(additive_mode_L ? probe_in_L : (a * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (a * probe_in_R));
			float_4 summed_probe_input_b = 
					(additive_mode_L ? probe_in_L : (b * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (b * probe_in_R));
			// Schrodinger equation, with added diffusion and decay
			// "multiplicative mode" should be considered the physically correct default here
			delta_a[i] =  (-summed_probe_input_a + - laplacian_b[i] - decay * a + safe_timestep * damping * laplacian_a[i]);
			delta_b[i] = ( summed_probe_input_b + laplacian_a[i] - decay * b + safe_timestep * damping * laplacian_b[i]);
		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void stepRK4Advection(		
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		std::vector<float_4> &dc_bias_a, std::vector<float_4> &dc_bias_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
			
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		float_4 probe_in_R = simd::rescale(t_input_R,-10.0,10.0,-1.0,1.0);
		float pos_to_mod_offset = 2.0*(pos_in_R - (MAX_POSITION / 2.0)) / MAX_POSITION;
		probe_in_R = sig_in_R * (probe_in_R + pos_to_mod_offset);

		float safe_timestep = getSafeTimestep();

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			float_4 probe_in_L = t_input_L * input_probe_L_window[i];
			
			float_4 a = a0[i];

			probe_out_L += a * output_probe_L_window[i];
			probe_out_R += a * output_probe_R_window[i];

			// additive mode necessarily works differently here
			float_4 summed_probe_input_a = 
					(additive_mode_L ? probe_in_L : (a * probe_in_L));
			
			// no multiplicative mode here for right probe, not because 
			// we can't do it, but because it's incredibly unstable
			//float_4 summed_probe_input_b = 
			//		(additive_mode_R ? probe_in_R : (a * probe_in_R));

			// gradients are smoothclamped for stability
			delta_a[i] = smoothclamp((summed_probe_input_a + safe_timestep * damping * laplacian_a[i] - decay * a)  // input
					- probe_in_R * gradient_a[i], -10.0f, 10.0f); // advection

		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void setParams(float damping, float timestep, float decay, float low_cut, float feedback) {
		this->damping = damping;
		this->timestep = timestep;
		this->decay = decay;
		this->low_cut = low_cut;
		this->feedback = feedback;
	}

	bool input_probe_L_dirty = true;
	bool input_probe_R_dirty = true;
	bool output_probe_L_dirty = true;
	bool output_probe_R_dirty = true;
	bool dirty_init = true;

	void setDirtyProbe(bool& dirty_flag, const float& pos_prev, const float& sig_prev, const float& pos_next, const float& sig_next) {
		dirty_flag = (pos_prev != pos_next || sig_prev != sig_next);
	}

	// Generate and normalize probe window buffers
	void generateProbeWindow(std::vector<float_4> &w, bool isDirty, float pos, float sigma, ProbeType probeType) {
		if (isDirty || dirty_init) {
			float_4 w_sum = float_4(0.0);
			for (int i = 0; i < CHANNEL_SIZE; i++) {
				float_4 f_i = float_4(4.0*i, 4.0*i+1.0, 4.0*i+2.0, 4.0*i+3.0);

				switch(probeType) {
					case ProbeType::INTEGRAL:
						w[i] = approxGaussian(pos, f_i, sigma);
						w_sum += w[i];
						break;
					case ProbeType::DIFFERENTIAL:
						w[i] = approxGaussianDeriv(pos, f_i, sigma);
						w_sum += simd::abs(w[i]);
						break;
					case ProbeType::SINC:
						w[i] = sinc(pos, f_i, sigma);
						w_sum += simd::abs(w[i]);
						break;
				}
			}
			float w_norm = sum(w_sum);
			if (probeType == ProbeType::SINC) {
				w_norm *= 0.5; //not the correct factor probably, but close
			}
			for (int i = 0; i < CHANNEL_SIZE; i++) {
				w[i] /= w_norm;
			}

			#ifdef DEBUG_PROBE_PRINT
				// WARNING! This will generate a HUGE amount of data in the log file 
				// when knobs are turned, or CV input is connected to the knob position/sigma
				std::string debug_string;
				for (auto f : w) {
					debug_string += std::to_string(f[0]) + " " + std::to_string(f[1]) + " " + std::to_string(f[2]) + " " + std::to_string(f[3]) + " ";
				}
				debug_string = "probe_generated: " + debug_string;
				INFO(debug_string.c_str());
			#endif
			dirty_init = false;
		}
	}

	// Probe window buffers are only updated when the inputs change to save on computation cost.
	void setProbeSettings(float pos_in_L, float pos_in_R, float pos_out_L, float pos_out_R, float sig_in_L, float sig_in_R, float sig_out_L, float sig_out_R) {
		setDirtyProbe(input_probe_L_dirty, this->pos_in_L, this->sig_in_L, pos_in_L, sig_in_L);
		setDirtyProbe(input_probe_R_dirty, this->pos_in_R, this->sig_in_R, pos_in_R, sig_in_R);
		setDirtyProbe(output_probe_L_dirty, this->pos_out_L, this->sig_out_L, pos_out_L, sig_out_L);
		setDirtyProbe(output_probe_R_dirty, this->pos_out_R, this->sig_out_R, pos_out_R, sig_out_R);
		this->pos_in_L = pos_in_L;
		this->pos_in_R = pos_in_R;
		this->sig_in_L = sig_in_L;
		this->sig_in_R = sig_in_R;
		this->pos_out_L = pos_out_L;
		this->pos_out_R = pos_out_R;
		this->sig_out_L = sig_out_L;
		this->sig_out_R = sig_out_R;
		generateProbeWindow(input_probe_L_window, input_probe_L_dirty, this->pos_in_L, this->sig_in_L, input_probe_type_L);
		generateProbeWindow(input_probe_R_window, input_probe_R_dirty, this->pos_in_R, this->sig_in_R, input_probe_type_R);
		generateProbeWindow(output_probe_L_window, output_probe_L_dirty, this->pos_out_L, this->sig_out_L, output_probe_type_L);
		generateProbeWindow(output_probe_R_window, output_probe_R_dirty, this->pos_out_R, this->sig_out_R, output_probe_type_R);
	}

	void toggleAdditiveModeL() {
		additive_mode_L = !additive_mode_L;
	}

	void toggleAdditiveModeR() {
		additive_mode_R = !additive_mode_R;
	}

	void updateProbeType(ProbeType &typeToChange) {
		switch(typeToChange) {
			case INTEGRAL:
				typeToChange = ProbeType::DIFFERENTIAL; break;
			case DIFFERENTIAL:
				typeToChange = ProbeType::SINC; break;
			case SINC:
				typeToChange = ProbeType::INTEGRAL; break;
		}
	}

	//void toggleDifferentialModeL() {
	void toggleInputProbeTypeL() {
		updateProbeType(input_probe_type_L);
		generateProbeWindow(input_probe_L_window, true, this->pos_in_L, this->sig_in_L, input_probe_type_L);
	}

	void toggleInputProbeTypeR() {
		updateProbeType(input_probe_type_R);
		generateProbeWindow(input_probe_R_window, true, this->pos_in_R, this->sig_in_R, input_probe_type_R);
	}

	void toggleOutputProbeTypeL() {
		updateProbeType(output_probe_type_L);
		generateProbeWindow(output_probe_L_window, true, this->pos_out_L, this->sig_out_L, output_probe_type_L);
	}

	void toggleOutputProbeTypeR() {
		updateProbeType(output_probe_type_R);
		generateProbeWindow(output_probe_R_window, true, this->pos_out_R, this->sig_out_R, output_probe_type_R);
	}


	void setProbeInputs(float amp_in_L, float amp_in_R) {
		this->amp_in_prev_L = this->amp_in_L;
		this->amp_in_prev_R = this->amp_in_R;
		this->amp_in_L = amp_in_L;
		this->amp_in_R = amp_in_R;
	}

	void setNextModel() {
		switch(this->model) {
			case WAVE_EQUATION:
				this->model = Model::SQUID_AXON;
				break;
			case SQUID_AXON:
				this->model = Model::SCHRODINGER;
				break;
			case SCHRODINGER:
				this->model = Model::RK4_ADVECTION;
				break;
			case RK4_ADVECTION:
				this->model = Model::WAVE_EQUATION;
				break;
		}
		setModelPointer();
	}

	void setModelPointer() {
		switch(this->model) {
			case SQUID_AXON:
				this->modelPointer = &WaveChannel::stepSquidAxon;
				break;
			case SCHRODINGER:
				this->modelPointer = &WaveChannel::stepSchrodinger;
				break;
			case RK4_ADVECTION:
				this->modelPointer = &WaveChannel::stepRK4Advection;
				break;
			case WAVE_EQUATION:
				this->modelPointer = &WaveChannel::stepWaveEquation;
				break;
		}
	}

	const char* getModelString() {
		const char* text;
		switch(model) {
			case WaveChannel::Model::WAVE_EQUATION:
				text = "WAVE_EQUATION";
				break;
			case WaveChannel::Model::SCHRODINGER:
				text = "SCHRODINGER";
				break;
			case WaveChannel::Model::RK4_ADVECTION:
				text = "RUNGE_KUTTA_RK4";
				break;
			case WaveChannel::Model::SQUID_AXON:
				text = "SQUID_AXON";
				break;
			default:
				text = "";
				break;
		}
		return text;
	}

	bool isModMode() {
		switch(model) {
			case WaveChannel::Model::RK4_ADVECTION:
				return true;
			default:
				return false;
		}
	}

	float getAmpOutL() {
		return amp_out_L;
	}

	float getAmpOutR() {
		return amp_out_R;
	}

	void setClipRange() {
		switch(clip_range_mode) {
			case ClipRange::V_10:
				clip_range = 10.0f; break;
			case ClipRange::V_30:
				clip_range = 30.0f; break;
			case ClipRange::V_60:
				clip_range = 60.0f; break;
			case ClipRange::V_100:
				clip_range = 100.0f; break;
			default:
				clip_range = 30.0f; break;
		}
	}

	// Update the ping-pong buffers
	void update() {
		setClipRange();
		setModelPointer();
		if (pong) {
			RK4_iter_3_8s(v_a0, v_b0, v_a1, v_b1);
		} else {
			RK4_iter_3_8s(v_a1, v_b1, v_a0, v_b0);
		}
		pong = !pong;
	}

};

struct WaterTable : Module {
	enum ParamIds {
		MODEL_BUTTON_PARAM,
		MULTIPLICATIVE_BUTTON_L_PARAM,
		MULTIPLICATIVE_BUTTON_R_PARAM,
		INPUT_PROBE_TYPE_BUTTON_L_PARAM,
		INPUT_PROBE_TYPE_BUTTON_R_PARAM,
		OUTPUT_PROBE_TYPE_BUTTON_L_PARAM,
		OUTPUT_PROBE_TYPE_BUTTON_R_PARAM,
		POSITION_IN_L_CV_PARAM,
		POSITION_IN_R_CV_PARAM,
		POSITION_IN_L_PARAM,
		POSITION_IN_R_PARAM,
		POSITION_OUT_L_CV_PARAM,
		POSITION_OUT_R_CV_PARAM,
		POSITION_OUT_L_PARAM,
		POSITION_OUT_R_PARAM,
		PROBE_SIGMA_IN_L_PARAM,
		PROBE_SIGMA_IN_R_PARAM,
		PROBE_SIGMA_IN_L_CV_PARAM,
		PROBE_SIGMA_IN_R_CV_PARAM,
		PROBE_SIGMA_OUT_L_PARAM,
		PROBE_SIGMA_OUT_L_CV_PARAM,
		PROBE_SIGMA_OUT_R_PARAM,
		PROBE_SIGMA_OUT_R_CV_PARAM,
		INPUT_GAIN_L_PARAM,
		INPUT_GAIN_L_CV_PARAM,
		INPUT_GAIN_R_PARAM,
		INPUT_GAIN_R_CV_PARAM,
		WET_PARAM,
		DRY_PARAM,
		TIMESTEP_PARAM,
		LOW_CUT_PARAM,
		DAMPING_PARAM,
		DECAY_PARAM,
		FEEDBACK_PARAM,
		WET_CV_PARAM,
		DRY_CV_PARAM,
		LOW_CUT_CV_PARAM,
		DAMPING_CV_PARAM,
		DECAY_CV_PARAM,
		FEEDBACK_CV_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		PROBE_IN_L_INPUT,
		PROBE_IN_R_INPUT,
		PROBE_POSITION_IN_L_INPUT,
		PROBE_POSITION_IN_R_INPUT,
		PROBE_SIGMA_IN_R_INPUT,
		PROBE_SIGMA_IN_L_INPUT,
		PROBE_POSITION_OUT_L_INPUT,
		PROBE_POSITION_OUT_R_INPUT,
		PROBE_SIGMA_OUT_L_INPUT,
		PROBE_SIGMA_OUT_R_INPUT,
		INPUT_GAIN_L_INPUT,
		INPUT_GAIN_R_INPUT,
		WET_INPUT,
		DRY_INPUT,
		LOW_CUT_INPUT,
		DAMPING_INPUT,
		DECAY_INPUT,
		FEEDBACK_INPUT,
		TIMESTEP_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		PROBE_OUT_L_OUTPUT,
		PROBE_OUT_R_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		EOC_LIGHT,
		POS_MODE_LIGHT,
		MOD_MODE_LIGHT,
		INTEGRAL_INPUT_L_LIGHT,
		INTEGRAL_INPUT_R_LIGHT,
		ADDITIVE_L_LIGHT,
		ADDITIVE_R_LIGHT,
		DIFFERENTIAL_INPUT_L_LIGHT,
		DIFFERENTIAL_INPUT_R_LIGHT,
		SINC_INPUT_L_LIGHT,
		SINC_INPUT_R_LIGHT,
		MULTIPLICATIVE_L_LIGHT,
		MULTIPLICATIVE_R_LIGHT,
		INTEGRAL_OUTPUT_L_LIGHT,
		INTEGRAL_OUTPUT_R_LIGHT,
		DIFFERENTIAL_OUTPUT_L_LIGHT,
		DIFFERENTIAL_OUTPUT_R_LIGHT,
		SINC_OUTPUT_L_LIGHT,
		SINC_OUTPUT_R_LIGHT,
		NUM_LIGHTS
	};

	WaveChannel waveChannel;
	StereoDCBiasRemover dcBias;
	FixedTimeExpSlewLimiter timestepSlewLimiter;
	dsp::ClockDivider lightDivider;

	CVParamInput<POSITION_IN_L_PARAM,  PROBE_POSITION_IN_L_INPUT,  POSITION_IN_L_CV_PARAM> pos_in_L_param;
	CVParamInput<POSITION_IN_R_PARAM,  PROBE_POSITION_IN_R_INPUT,  POSITION_IN_R_CV_PARAM> pos_in_R_param;
	CVParamInput<POSITION_OUT_L_PARAM, PROBE_POSITION_OUT_L_INPUT, POSITION_OUT_L_CV_PARAM> pos_out_L_param;
	CVParamInput<POSITION_OUT_R_PARAM, PROBE_POSITION_OUT_R_INPUT, POSITION_OUT_R_CV_PARAM> pos_out_R_param;

	CVParamInput<PROBE_SIGMA_IN_L_PARAM, PROBE_SIGMA_IN_L_INPUT, PROBE_SIGMA_IN_L_CV_PARAM> sig_in_L_param;
	CVParamInput<PROBE_SIGMA_IN_R_PARAM, PROBE_SIGMA_IN_R_INPUT, PROBE_SIGMA_IN_R_CV_PARAM> sig_in_R_param;
	CVParamInput<PROBE_SIGMA_OUT_L_PARAM, PROBE_SIGMA_OUT_L_INPUT, PROBE_SIGMA_OUT_L_CV_PARAM> sig_out_L_param;
	CVParamInput<PROBE_SIGMA_OUT_R_PARAM, PROBE_SIGMA_OUT_R_INPUT, PROBE_SIGMA_OUT_R_CV_PARAM> sig_out_R_param;

	CVParamInput<INPUT_GAIN_L_PARAM,  INPUT_GAIN_L_INPUT,   INPUT_GAIN_L_CV_PARAM> input_gain_L_param;
	CVParamInput<INPUT_GAIN_R_PARAM,  INPUT_GAIN_R_INPUT,   INPUT_GAIN_R_CV_PARAM> input_gain_R_param;

	CVParamInput<DAMPING_PARAM,  DAMPING_INPUT,   DAMPING_CV_PARAM> damping_param;
	CVParamInput<TIMESTEP_PARAM, TIMESTEP_INPUT,  DUMMY_CV> timestep_param;
	CVParamInput<DECAY_PARAM,    DECAY_INPUT,     DECAY_CV_PARAM> decay_param;
	CVParamInput<FEEDBACK_PARAM, FEEDBACK_INPUT,  FEEDBACK_CV_PARAM> feedback_param;
	CVParamInput<LOW_CUT_PARAM,  LOW_CUT_INPUT,   LOW_CUT_CV_PARAM> low_cut_param;
	CVParamInput<DRY_PARAM,  DRY_INPUT,   DRY_CV_PARAM> dry_param;
	CVParamInput<WET_PARAM,  WET_INPUT,   WET_CV_PARAM> wet_param;
	
	#define PROBE_SIGMA_MIN 0.25
	#define PROBE_SIGMA_MAX 4.0
	#define PROBE_SIGMA_DEF 1.0

	#define LOW_CUT_MIN 0.0
	#define LOW_CUT_MAX 1.0
	#define LOW_CUT_DEF 0.0

	#define DAMPING_MIN 0.0
	#define DAMPING_MAX 0.95
	#define DAMPING_DEF 0.5

	#define DECAY_MIN 0.0
	#define DECAY_MAX 0.5
	#define DECAY_DEF 0.002

	//#define TIMESTEP_SHIFT 3.191
	#define TIMESTEP_SHIFT 5.191
	#define TIMESTEP_MAX 0.4
	#define TIMESTEP_KNOB_MIN -5.0
	#define TIMESTEP_KNOB_MAX 5.0
	#define TIMESTEP_DEF 0.0
	#define TIMESTEP_SLEW_RATE 0.02f
	#define TIMESTEP_POST_SCALE 1.5

	#define FEEDBACK_MIN -8.0
	#define FEEDBACK_MAX 8.0
	#define FEEDBACK_DEF 0.0

	#define MIN_GAIN 0.0
	#define MAX_GAIN 8.0
	#define DEF_GAIN 6.0

	
	WaterTable() : timestepSlewLimiter(TIMESTEP_SLEW_RATE) {
		
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		pos_in_L_param.configModulo(this, MAX_POSITION, 0.0, "pos_in_L", "Left Input Probe Position");
		pos_out_L_param.configModulo(this, MAX_POSITION, 0.5, "pos_out_L", "Left Output Probe Position");
		pos_in_R_param.configModulo(this, MAX_POSITION, 1.0, "pos_in_R", "Right Input Probe Position");
		pos_out_R_param.configModulo(this, MAX_POSITION, 1.5, "pos_out_R", "Right Output Probe Position");
		sig_in_L_param.config(this, PROBE_SIGMA_MIN, PROBE_SIGMA_MAX, PROBE_SIGMA_DEF, "sig_in_L", "Left Input Probe Width");
		sig_out_L_param.config(this, PROBE_SIGMA_MIN, PROBE_SIGMA_MAX, PROBE_SIGMA_DEF, "sig_out_L", "Left Output Probe Width");
		sig_in_R_param.config(this, PROBE_SIGMA_MIN, PROBE_SIGMA_MAX, PROBE_SIGMA_DEF, "sig_in_R", "Right Input Probe Width");
		sig_out_R_param.config(this, PROBE_SIGMA_MIN, PROBE_SIGMA_MAX, PROBE_SIGMA_DEF, "sig_out_R", "Right Output Probe Width");
		damping_param.configExp(this, DAMPING_MIN, DAMPING_MAX, DAMPING_DEF, "damping", "Damping");
		timestep_param.configPitch(this, TIMESTEP_POST_SCALE, 1.0, TIMESTEP_SHIFT, TIMESTEP_KNOB_MIN, TIMESTEP_KNOB_MAX, TIMESTEP_MAX, TIMESTEP_DEF, "timestep", "Timestep");
		decay_param.configExp(this, DECAY_MIN, DECAY_MAX, DECAY_DEF, "decay", "Decay");
		feedback_param.configBipolarExp(this, FEEDBACK_MIN, FEEDBACK_MAX, FEEDBACK_DEF, "feedback", "Feedback");
		low_cut_param.config(this, LOW_CUT_MIN, LOW_CUT_MAX, LOW_CUT_DEF, "low_cut", "Low Cut");
		input_gain_L_param.configExp(this, MIN_GAIN , MAX_GAIN, DEF_GAIN, "input_gain_L", "Input Gain L");
		input_gain_R_param.configExp(this, MIN_GAIN , MAX_GAIN, DEF_GAIN, "input_gain_R", "Input Gain R");
		dry_param.configExp(this, MIN_GAIN , MAX_GAIN, MIN_GAIN , "dry", "Dry Gain");
		wet_param.configExp(this, MIN_GAIN , MAX_GAIN, DEF_GAIN, "wet", "Wet Gain");
		configOutput(PROBE_OUT_L_OUTPUT, "Left");
		configOutput(PROBE_OUT_R_OUTPUT, "Right");
		configInput(PROBE_IN_L_INPUT, "Left");
		configInput(PROBE_IN_R_INPUT, "Right");
		lightDivider.setDivision(16);
	}

	bool anyOutputsConnected() {
		return outputs[PROBE_OUT_L_OUTPUT].isConnected() || outputs[PROBE_OUT_R_OUTPUT].isConnected();
	}

	void setNextModel() {
		waveChannel.setNextModel();
	}

	void setLightPatternProbeType(WaveChannel::ProbeType probeType, float &integralLight, float &differentialLight, float &sincLight, bool override) {
		switch(probeType) {
			case WaveChannel::ProbeType::DIFFERENTIAL:
				differentialLight = 1.0;
				integralLight = 0.0;
				sincLight = 0.0;
				break;
			case WaveChannel::ProbeType::INTEGRAL:
				differentialLight = 0.0;
				integralLight = 1.0;
				sincLight = 0.0;
				break;
			case WaveChannel::ProbeType::SINC:
				differentialLight = 0.0;
				integralLight = 0.0;
				sincLight = 1.0;
				break;
			default:
				differentialLight = 0.0;
				integralLight = 0.0;
				sincLight = 0.0;
				break;
		}
		if (override) {
			differentialLight = 0.0;
			integralLight = 0.0;
			sincLight = 0.0;
		}
	}

	void setLightPatternAdditive(bool additiveMode, float &additiveLight, float &multiplicativeLight, bool override) {
		if (override) {
			additiveLight = 0.0;
			multiplicativeLight = 0.0;
		} else {
			additiveLight = additiveMode ? 1.0 : 0.0;
			multiplicativeLight = additiveMode ? 0.0 : 1.0;
		}
	}

	void onReset() override {
		waveChannel.additive_mode_L = true;
		waveChannel.additive_mode_R = true;
		waveChannel.input_probe_type_L = WaveChannel::ProbeType::INTEGRAL;
		waveChannel.input_probe_type_R = WaveChannel::ProbeType::INTEGRAL;
		waveChannel.output_probe_type_L = WaveChannel::ProbeType::INTEGRAL;
		waveChannel.output_probe_type_R = WaveChannel::ProbeType::INTEGRAL;
		waveChannel.dirty_init = true;
	}

	void process(const ProcessArgs& args) override {
		float pos_in_L = pos_in_L_param.getValue();
		float pos_in_R = pos_in_R_param.getValue();
		float sig_in_L = sig_in_L_param.getValue();
		float sig_in_R = sig_in_R_param.getValue();
		float pos_out_L = pos_out_L_param.getValue();
		float pos_out_R = pos_out_R_param.getValue();
		float sig_out_L = sig_out_L_param.getValue();
		float sig_out_R = sig_out_R_param.getValue();
		float damping = damping_param.getValue();
		float decay = decay_param.getValue();
		float feedback = feedback_param.getValue();
		float low_cut = low_cut_param.getValue();

		float sample_rate_scale = 96000.0f / args.sampleRate;
		timestep_param.setSampleRateScale(sample_rate_scale);
		float timestep = timestep_param.getValue();

		// volume is basically proportional to the square root of timestep,
		// so we slew limit timestep to prevent clicking, 
		// and later divide the final amplitude through by sqrt(timestep)
		timestepSlewLimiter.limit(timestep);

		float amp_in_L = inputs[PROBE_IN_L_INPUT].getVoltage(0);
		float amp_in_R = inputs[PROBE_IN_R_INPUT].getVoltage(0);

		float amp_out_L = 0.;
		float amp_out_R = 0.;
		if (anyOutputsConnected()) {
			waveChannel.setParams(damping, timestep, decay, low_cut, feedback);
			waveChannel.setProbeSettings(pos_in_L, pos_in_R, pos_out_L, pos_out_R, sig_in_L, sig_in_R, sig_out_L, sig_out_R);
			waveChannel.setProbeInputs(input_gain_L_param.getValue() * amp_in_L, input_gain_R_param.getValue() * amp_in_R);
			waveChannel.update();
			amp_out_L = waveChannel.getAmpOutL();
			amp_out_R = waveChannel.getAmpOutR();
			dcBias.remove(amp_out_L, amp_out_R);
			float ts_curved = simd::sqrt(timestep);
			amp_out_L *= (wet_param.getValue() / ts_curved);
			amp_out_R *= (wet_param.getValue() / ts_curved);
		}

		if (outputs[PROBE_OUT_L_OUTPUT].isConnected()) {
			outputs[PROBE_OUT_L_OUTPUT].setVoltage(dry_param.getValue() * amp_in_L + amp_out_L, 0);
		}

		if (outputs[PROBE_OUT_R_OUTPUT].isConnected()) {
			outputs[PROBE_OUT_R_OUTPUT].setVoltage(dry_param.getValue() * amp_in_R + amp_out_R, 0);
		}
		

		outputs[PROBE_OUT_L_OUTPUT].setChannels(1);
		outputs[PROBE_OUT_R_OUTPUT].setChannels(1);


		// Light
		if (lightDivider.process()) {
				float lightValue = amp_out_L;
				lights[EOC_LIGHT].setSmoothBrightness(lightValue, args.sampleTime * lightDivider.getDivision());

				float pos_light = 0.0;
				float mod_light = 0.0;
				bool disable_R_diff_add_lights = false;
				switch(waveChannel.model) {
					case WaveChannel::Model::RK4_ADVECTION:
						pos_light = 0.0; mod_light = 1.0; 
						disable_R_diff_add_lights = true;
						break;
					default:
						pos_light = 1.0; mod_light = 0.0; 
						disable_R_diff_add_lights = false;
						break;
				}

				float input_diff_light_l, input_int_light_l, input_sinc_light_l, input_diff_light_r, input_int_light_r, input_sinc_light_r;
				float output_diff_light_l, output_int_light_l, output_sinc_light_l, output_diff_light_r, output_int_light_r, output_sinc_light_r;
				float add_light_l, mult_light_l, add_light_r, mult_light_r;

				// TODO: make this more DRY
				setLightPatternProbeType(waveChannel.input_probe_type_L, input_int_light_l, input_diff_light_l, input_sinc_light_l, false);
				setLightPatternProbeType(waveChannel.input_probe_type_R, input_int_light_r, input_diff_light_r, input_sinc_light_r, disable_R_diff_add_lights);
				setLightPatternProbeType(waveChannel.output_probe_type_L, output_int_light_l, output_diff_light_l, output_sinc_light_l, false);
				setLightPatternProbeType(waveChannel.output_probe_type_R, output_int_light_r, output_diff_light_r, output_sinc_light_r, false);

				setLightPatternAdditive(waveChannel.additive_mode_L, add_light_l, mult_light_l, false);
				setLightPatternAdditive(waveChannel.additive_mode_R, add_light_r, mult_light_r, disable_R_diff_add_lights);

				lights[POS_MODE_LIGHT].setBrightness(pos_light);
				lights[MOD_MODE_LIGHT].setBrightness(mod_light);

				lights[DIFFERENTIAL_INPUT_L_LIGHT].setBrightness(input_diff_light_l);
				lights[INTEGRAL_INPUT_L_LIGHT].setBrightness(input_int_light_l);
				lights[SINC_INPUT_L_LIGHT].setBrightness(input_sinc_light_l);
				lights[DIFFERENTIAL_INPUT_R_LIGHT].setBrightness(input_diff_light_r);
				lights[INTEGRAL_INPUT_R_LIGHT].setBrightness(input_int_light_r);
				lights[SINC_INPUT_R_LIGHT].setBrightness(input_sinc_light_r);

				lights[DIFFERENTIAL_OUTPUT_L_LIGHT].setBrightness(output_diff_light_l);
				lights[INTEGRAL_OUTPUT_L_LIGHT].setBrightness(output_int_light_l);
				lights[SINC_OUTPUT_L_LIGHT].setBrightness(output_sinc_light_l);
				lights[DIFFERENTIAL_OUTPUT_R_LIGHT].setBrightness(output_diff_light_r);
				lights[INTEGRAL_OUTPUT_R_LIGHT].setBrightness(output_int_light_r);
				lights[SINC_OUTPUT_R_LIGHT].setBrightness(output_sinc_light_r);

				lights[ADDITIVE_L_LIGHT].setBrightness(add_light_l);
				lights[MULTIPLICATIVE_L_LIGHT].setBrightness(mult_light_l);
				lights[ADDITIVE_R_LIGHT].setBrightness(add_light_r);
				lights[MULTIPLICATIVE_R_LIGHT].setBrightness(mult_light_r);

		}
	}

	int getOversamplingMode() {
		return static_cast<int>(waveChannel.oversampling_mode);
	}

	void setOversamplingMode(int mode) {
		waveChannel.oversampling_mode = static_cast<WaveChannel::OversamplingMode>(mode);
	}

	int getClipRangeMode() {
		return static_cast<int>(waveChannel.clip_range_mode);
	}

	void setClipRangeMode(int mode) {
		waveChannel.clip_range_mode = static_cast<WaveChannel::ClipRange>(mode);
	}

	void booleanFromJson(json_t* rootJ, bool &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = json_boolean_value(j_val);
	}

	void booleanToJson(json_t* rootJ, bool &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_boolean(val));
	}

	void modelFromJson(json_t* rootJ, WaveChannel::Model &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel::Model>(json_integer_value(j_val));
	}

	void modelToJson(json_t* rootJ, WaveChannel::Model &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void probeFromJson(json_t* rootJ, WaveChannel::ProbeType &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel::ProbeType>(json_integer_value(j_val));
	}

	void probeToJson(json_t* rootJ, WaveChannel::ProbeType &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void oversamplingModeFromJson(json_t* rootJ, WaveChannel::OversamplingMode &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel::OversamplingMode>(json_integer_value(j_val));
	}

	void oversamplingModeToJson(json_t* rootJ, WaveChannel::OversamplingMode &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void clipRangeModeFromJson(json_t* rootJ, WaveChannel::ClipRange &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel::ClipRange>(json_integer_value(j_val));
	}

	void clipRangeModeToJson(json_t* rootJ, WaveChannel::ClipRange &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	json_t* dataToJson() override {
		json_t* rootJ = json_object();
		pos_in_L_param.dataToJson(rootJ);
		pos_in_R_param.dataToJson(rootJ);
		pos_out_L_param.dataToJson(rootJ);
		pos_out_R_param.dataToJson(rootJ);
		probeToJson(rootJ, waveChannel.input_probe_type_L, "input_probe_type_L");
		probeToJson(rootJ, waveChannel.input_probe_type_R, "input_probe_type_R");
		probeToJson(rootJ, waveChannel.output_probe_type_L, "output_probe_type_L");
		probeToJson(rootJ, waveChannel.output_probe_type_R, "output_probe_type_R");
		booleanToJson(rootJ, waveChannel.additive_mode_L, "additive_mode_L");
		booleanToJson(rootJ, waveChannel.additive_mode_R, "additive_mode_R");
		modelToJson(rootJ, waveChannel.model, "model");
		oversamplingModeToJson(rootJ, waveChannel.oversampling_mode, "oversampling_mode");
		clipRangeModeToJson(rootJ, waveChannel.clip_range_mode, "clip_range_mode");
		return rootJ;
	}

	void dataFromJson(json_t* rootJ) override {
		pos_in_L_param.dataFromJson(rootJ);
		pos_in_R_param.dataFromJson(rootJ);
		pos_out_L_param.dataFromJson(rootJ);
		pos_out_R_param.dataFromJson(rootJ);
		probeFromJson(rootJ, waveChannel.input_probe_type_L, "input_probe_type_L");
		probeFromJson(rootJ, waveChannel.input_probe_type_R, "input_probe_type_R");
		probeFromJson(rootJ, waveChannel.output_probe_type_L, "output_probe_type_L");
		probeFromJson(rootJ, waveChannel.output_probe_type_R, "output_probe_type_R");
		booleanFromJson(rootJ, waveChannel.additive_mode_L, "additive_mode_L");
		booleanFromJson(rootJ, waveChannel.additive_mode_R, "additive_mode_R");
		modelFromJson(rootJ, waveChannel.model, "model");
		oversamplingModeFromJson(rootJ, waveChannel.oversampling_mode, "oversampling_mode");
		clipRangeModeFromJson(rootJ, waveChannel.clip_range_mode, "clip_range_mode");
	}

	void onReset(const ResetEvent& e) override {
		pos_in_L_param.reset();
		pos_in_R_param.reset();
		pos_out_L_param.reset();
		pos_out_R_param.reset();
		Module::onReset(e);
	}

	void onRandomize(const RandomizeEvent& e) override {
		pos_in_L_param.randomize();
		pos_in_R_param.randomize();
		pos_out_L_param.randomize();
		pos_out_R_param.randomize();
		Module::onRandomize(e);
	}
};


struct WaterTableWidget : ModuleWidget {
	WaterTableWidget(WaterTable* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/WaterTable.svg")));

		/*  lambdas below MUST BE pass by value or we segfault when the reference goes out of scope.
			this way of setting up the buttons is somewhat bizarre, but it does reduce boilerplate substantially.
		*/
		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(69.566, 83.327)), module, WaterTable::MODEL_BUTTON_PARAM));
			FreeSurfaceLogoToggleDark<WaterTable, 4>* button 
					= createParamCentered<FreeSurfaceLogoToggleDark<WaterTable, 4>>(mm2px(Vec(69.566, 83.327)), module, WaterTable::MODEL_BUTTON_PARAM);
			button->config(
				"Model", 
				std::vector<std::string>{"WAVE EQUATION", "SQUID AXON", "SCHRODINGER", "RUNGE KUTTA RK4"},
				true, 
				[=] () -> int { return static_cast<int>(module->waveChannel.model); }, 
				[=] () -> void { module->waveChannel.setNextModel(); }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(18.94, 66.2)), module, WaterTable::MULTIPLICATIVE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable, 2>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 2>>(mm2px(Vec(18.94, 66.2)), module, WaterTable::MULTIPLICATIVE_BUTTON_L_PARAM);
			button->config(
				"Left Input Mode",
				std::vector<std::string>{"MULTIPLICATIVE", "ADDITIVE"},
				true, 
				[=] () -> int { return module->waveChannel.additive_mode_L ? 1 : 0; }, 
				[=] () -> void { module->waveChannel.toggleAdditiveModeL(); }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(48.062, 66.2)), module, WaterTable::MULTIPLICATIVE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 3>>(mm2px(Vec(48.062, 66.2)), module, WaterTable::MULTIPLICATIVE_BUTTON_R_PARAM);
			button->config(
				"Right Input Mode",
				std::vector<std::string>{"MULTIPLICATIVE", "ADDITIVE", "DISABLED"},
				true, 
				[=] () -> int { return module->waveChannel.isModMode() ? 2 : (module->waveChannel.additive_mode_R ? 1 : 0); }, 
				[=] () -> void { if (!module->waveChannel.isModMode()) { module->waveChannel.toggleAdditiveModeR(); } }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(5.018, 66.198)), module, WaterTable::INPUT_PROBE_TYPE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 3>>(mm2px(Vec(5.018, 66.198)), module, WaterTable::INPUT_PROBE_TYPE_BUTTON_L_PARAM);
			button->config(
				"Left Input Shape",
				std::vector<std::string>{"INTEGRAL", "DIFFERENTIAL", "SINC"},
				true, 
				[=] () -> int { return static_cast<int>(module->waveChannel.input_probe_type_L); }, 
				[=] () -> void { module->waveChannel.toggleInputProbeTypeL(); }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(34.14, 66.198)), module, WaterTable::INPUT_PROBE_TYPE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable, 4>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 4>>(mm2px(Vec(34.14, 66.198)), module, WaterTable::INPUT_PROBE_TYPE_BUTTON_R_PARAM);
			button->config(
				"Right Input Shape",
				std::vector<std::string>{"INTEGRAL", "DIFFERENTIAL", "SINC", "DISABLED"},
				true, 
				[=] () -> int { return module->waveChannel.isModMode() ? 3 : static_cast<int>(module->waveChannel.input_probe_type_R); }, 
				[=] () -> void { if (!module->waveChannel.isModMode()) { module->waveChannel.toggleInputProbeTypeR(); } }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(4.991, 121.739)), module, WaterTable::OUTPUT_PROBE_TYPE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 3>>(mm2px(Vec(4.991, 121.739)), module, WaterTable::OUTPUT_PROBE_TYPE_BUTTON_L_PARAM);
			button->config(
				"Left Output Shape",
				std::vector<std::string>{"INTEGRAL", "DIFFERENTIAL", "SINC"},
				true, 
				[=] () -> int { return static_cast<int>(module->waveChannel.output_probe_type_L); }, 
				[=] () -> void { module->waveChannel.toggleOutputProbeTypeL(); }, 
				module
			);
			addParam(button);
		}

		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(34.287, 121.739)), module, WaterTable::OUTPUT_PROBE_TYPE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable, 3>>(mm2px(Vec(34.287, 121.739)), module, WaterTable::OUTPUT_PROBE_TYPE_BUTTON_R_PARAM);
			button->config(
				"Right Output Shape",
				std::vector<std::string>{"INTEGRAL", "DIFFERENTIAL", "SINC"},
				true, 
				[=] () -> int { return static_cast<int>(module->waveChannel.output_probe_type_R); }, 
				[=] () -> void { module->waveChannel.toggleOutputProbeTypeR(); }, 
				module
			);
			addParam(button);
		}

		{
			// mm2px(Vec(60.444, 62.491))
			//addChild(createWidget<Widget>(mm2px(Vec(59.822, 9.072))));
			WaterTableDisplay<WaterTable, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>* display = new WaterTableDisplay<WaterTable, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>();
			display->module = module;
			display->box.pos = mm2px(Vec(59.822, 9.072));
			display->box.size = mm2px(Vec(60.444, 62.491));
			display->setBBox();
			addChild(display);
		}
		
		

		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(23.75, 19.313)), module, WaterTable::POSITION_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(53.046, 19.313)), module, WaterTable::POSITION_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 40.993)), module, WaterTable::PROBE_SIGMA_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.717, 40.993)), module, WaterTable::PROBE_SIGMA_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 53.047)), module, WaterTable::INPUT_GAIN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.676, 53.047)), module, WaterTable::INPUT_GAIN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(23.75, 89.313)), module, WaterTable::POSITION_OUT_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(53.046, 89.313)), module, WaterTable::POSITION_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(83.969, 104.575)), module, WaterTable::DAMPING_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(94.324, 104.575)), module, WaterTable::DECAY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(104.848, 104.575)), module, WaterTable::FEEDBACK_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(115.382, 104.575)), module, WaterTable::LOW_CUT_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.676, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(95.75, 93.843)), module, WaterTable::DRY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(115.288, 93.843)), module, WaterTable::WET_CV_PARAM));

		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(10.25, 24.063)), module, WaterTable::POSITION_IN_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(39.546, 24.063)), module, WaterTable::POSITION_IN_R_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(10.25, 94.063)), module, WaterTable::POSITION_OUT_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(39.546, 94.063)), module, WaterTable::POSITION_OUT_R_PARAM));

		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(69.566, 109.631)), module, WaterTable::TIMESTEP_PARAM));
		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(110.538, 81.989)), module, WaterTable::WET_PARAM));
		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(91.0, 82.107)), module, WaterTable::DRY_PARAM));

		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 40.64)), module, WaterTable::PROBE_SIGMA_IN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 40.64)), module, WaterTable::PROBE_SIGMA_IN_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(83.969, 112.384)), module, WaterTable::DAMPING_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(94.324, 112.392)), module, WaterTable::DECAY_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(104.848, 112.392)), module, WaterTable::FEEDBACK_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(115.382, 112.392)), module, WaterTable::LOW_CUT_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 52.798)), module, WaterTable::INPUT_GAIN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 52.798)), module, WaterTable::INPUT_GAIN_R_PARAM));

		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.222, 7.244)), module, WaterTable::PROBE_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(44.296, 7.244)), module, WaterTable::PROBE_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 28.063)), module, WaterTable::PROBE_POSITION_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 28.063)), module, WaterTable::PROBE_POSITION_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 40.64)), module, WaterTable::PROBE_SIGMA_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 40.64)), module, WaterTable::PROBE_SIGMA_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 52.691)), module, WaterTable::INPUT_GAIN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 52.691)), module, WaterTable::INPUT_GAIN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(87.0, 93.843)), module, WaterTable::DRY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(106.538, 93.843)), module, WaterTable::WET_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 98.063)), module, WaterTable::PROBE_POSITION_OUT_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 98.063)), module, WaterTable::PROBE_POSITION_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 110.41)), module, WaterTable::PROBE_SIGMA_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(73.56, 120.947)), module, WaterTable::TIMESTEP_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(83.969, 120.947)), module, WaterTable::DAMPING_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(94.324, 120.947)), module, WaterTable::DECAY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(104.848, 120.947)), module, WaterTable::FEEDBACK_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(115.382, 120.947)), module, WaterTable::LOW_CUT_INPUT));

		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.222, 78.379)), module, WaterTable::PROBE_OUT_L_OUTPUT));
		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(44.296, 78.379)), module, WaterTable::PROBE_OUT_R_OUTPUT));


		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(118.795, 4.673)), module, WaterTable::EOC_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.628, 5.978)), module, WaterTable::POS_MODE_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.628, 8.295)), module, WaterTable::MOD_MODE_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(26.398, 63.259)), module, WaterTable::ADDITIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 63.278)), module, WaterTable::INTEGRAL_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 63.278)), module, WaterTable::INTEGRAL_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.52, 63.278)), module, WaterTable::ADDITIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 66.2)), module, WaterTable::DIFFERENTIAL_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 66.2)), module, WaterTable::DIFFERENTIAL_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 69.123)), module, WaterTable::SINC_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 69.123)), module, WaterTable::SINC_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.52, 69.123)), module, WaterTable::MULTIPLICATIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(26.398, 69.141)), module, WaterTable::MULTIPLICATIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(11.437, 123.322)), module, WaterTable::INTEGRAL_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(17.915, 123.322)), module, WaterTable::DIFFERENTIAL_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(24.393, 123.322)), module, WaterTable::SINC_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(40.383, 123.322)), module, WaterTable::INTEGRAL_OUTPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(46.861, 123.322)), module, WaterTable::DIFFERENTIAL_OUTPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(53.339, 123.322)), module, WaterTable::SINC_OUTPUT_R_LIGHT));

	}

	void appendContextMenu(Menu* menu) override {
		WaterTable* module = dynamic_cast<WaterTable*>(this->module);
		assert(module);

		menu->addChild(new MenuSeparator);
		//menu->addChild(createMenuLabel(""));

		menu->addChild(createIndexSubmenuItem("Oversampling mode",
			{"Sinc", "Biquad"},
			[=]() {
				return module->getOversamplingMode();
			},
			[=](int mode) {
				module->setOversamplingMode(mode);
			}
		));

		menu->addChild(createIndexSubmenuItem("Internal clip range",
			{"10V", "30V", "60V", "100V"},
			[=]() {
				return module->getClipRangeMode();
			},
			[=](int mode) {
				module->setClipRangeMode(mode);
			}
		));
	}
};


Model* modelWaterTable = createModel<WaterTable, WaterTableWidget>("FreeSurface-WaterTable");
