#include "plugin.hpp"
#include "OpCache.hpp"
#include "Profiler.hpp"
#include <math.h>
#include <cstring>
/** Enabling this resamples inputs and outputs from each
	RK4 iteration to get the final output.
	It would be nice to use this here, but it's way too expensive
	so for now it's hidden behind a define.
	In the mean time, we use zero-pad upsampling for the input
	and biquad lowpass on the output. */
// #define RESAMPLED_IO
#ifdef RESAMPLED_IO
	#include "VectorStores.hpp"
#endif

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

// Just disables quadratic bezier interpolation 
// so we can see the raw output better
//#define DRAW_DEBUG

struct WaveChannel {
	enum Model {
		WAVE_EQUATION,
		SQUID_AXON,
		SCHRODINGER,
		RK4_ADVECTION
	};
	Model model;

	dsp::BiquadFilter biquad_L;
	dsp::BiquadFilter biquad_R;

	#ifdef RESAMPLED_IO
	unique_resampler_stereo_float resampler_in = 
        std::unique_ptr<resampler_stereo_float>(
            new resampler_stereo_float(8, 2)
        );

	unique_resampler_stereo_float resampler_out = 
        std::unique_ptr<resampler_stereo_float>(
            new resampler_stereo_float(2, 8)
        );
	#endif

	        

	/** Member function pointer for the current model.
	 *  Nasty, but using a switch or inheritance would be nastier 
	 *  here and probably worse for performance.
	 */
	typedef void (WaveChannel::*ModelPointer) (
		const std::vector<float_4>&, const std::vector<float_4>&, 
		const std::vector<float_4>&, const std::vector<float_4>&,
		const std::vector<float_4>&, const std::vector<float_4>&,
		std::vector<float_4>&, std::vector<float_4>&,
		const float&, const float&,
		float&, float&);

	ModelPointer modelPointer;

	float pos_in_L, pos_in_R, amp_in_L, amp_in_R, amp_in_prev_L, amp_in_prev_R, sig_in_L, sig_in_R, pos_out_L, pos_out_R, sig_out_L, sig_out_R;
	float amp_out_L, amp_out_R = 0.0; 
	float damping = 0.1; 
	float timestep = 0.01;
	float decay = 0.005;
	float feedback = 0.0;
	float low_cut = 0.0;

	// ping pong buffer setup
	bool pong = false;

	bool differential_mode_L = true;
	bool differential_mode_R = true;
	bool additive_mode_L = true;
	bool additive_mode_R = true;

	std::vector<float_4> v_a0 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_b0 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_a1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> v_b1 = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

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
		biquad_L.setParameters(dsp::BiquadFilter::Type::LOWPASS, 0.25f, 0.5f, 1.f);
		biquad_R.setParameters(dsp::BiquadFilter::Type::LOWPASS, 0.25f, 0.5f, 1.f);

		#ifdef RESAMPLED_IO
		resampler_in->reset();
        resampler_in->finalize();
		resampler_in->setResampleRatio(4.0);
		resampler_out->reset();
		resampler_out->finalize();
		resampler_out->setResampleRatio(0.25);
		#endif
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

	inline void processSample(float &sample_L, float &sample_R) {
		sample_L = biquad_L.process(sample_L);
		sample_R = biquad_R.process(sample_R);
	}

	inline void modelIteration(
			const std::vector<float_4> &a_in, std::vector<float_4> &delta_a,
			const std::vector<float_4> &b_in, std::vector<float_4> &delta_b,
			float &input_L, float &input_R, 
			float &t_amp_out_L, float &t_amp_out_R) {

		gradient_and_laplacian(a_in, t_gradient_a, t_laplacian_a);
		gradient_and_laplacian(b_in, t_gradient_b, t_laplacian_b);
		(this->*modelPointer)(a_in, b_in, t_laplacian_a, t_laplacian_b, 
		t_gradient_a, t_gradient_b, delta_a, delta_b, 
		input_L, input_R, t_amp_out_L, t_amp_out_R);

		processSample(t_amp_out_L, t_amp_out_R);
	}

	#ifdef RESAMPLED_IO
	inline void prepareInput() {
		resampler_in->pushInput(amp_in_L, amp_out_R);
		resampler_in->resample();
	}

	float t_amp_in_prev_L;
	float t_amp_in_prev_R;

	inline void processInput(float &input_L, float &input_R, const float &feedback_amp_L, const float &feedback_amp_R) {
		StereoSample out = resampler_in->shiftOutput();
		float t_amp_in_L = out.x;
		float t_amp_in_R = out.y;
		input_L = feedback * feedback_amp_L + t_amp_in_L - low_cut * t_amp_in_prev_L;
		input_R = feedback * feedback_amp_R + t_amp_in_R - low_cut * t_amp_in_prev_R;
		t_amp_in_prev_L = out.x;
		t_amp_in_prev_R = out.y;
	}

	inline void processOutput(const float &output_L, const float &output_R) {
		resampler_out->pushInput(output_L, output_R);
	}

	inline StereoSample getResampledOutput() {
		resampler_out->resample();
		return resampler_out->shiftOutput();
	}
	#endif

	/** Runge-Kutta (RK4) integration,
	 *  Using the 3/8s Runge Kutta method.
	 *  Used to increase stability at large timesteps
	 *  and also to upsample our input.
	 */
	#define I_CLAMP 30.0
	#define F_CLAMP 10.0
	#define INTER_CLAMP(x) smoothclamp((x),-I_CLAMP,I_CLAMP)
	#define FINAL_CLAMP(x) smoothclamp((x),-F_CLAMP,F_CLAMP)
	void RK4_iter_3_8s(
			const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
			std::vector<float_4> &a1, std::vector<float_4> &b1) {

		float t_amp_out_L;
		float t_amp_out_R;

		const float third = 1.0/3.0;

		#ifdef RESAMPLED_IO
		prepareInput();
		#endif

		// Round 1, initial step
		float input_L = feedback * amp_out_L + amp_in_L - low_cut * amp_in_prev_L;
		float input_R = feedback * amp_out_R + amp_in_R - low_cut * amp_in_prev_R;

		//float input_L, input_R;
		#ifdef RESAMPLED_IO
		processInput(input_L, input_R, amp_out_L, amp_out_R);
		#endif

		modelIteration( a0, a_grad_1, b0, b_grad_1, input_L, input_R, t_amp_out_L, t_amp_out_R);

		#ifdef RESAMPLED_IO
		processOutput(t_amp_out_L, t_amp_out_R);
		#endif

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_2[i] = a0[i] + third * timestep * a_grad_1[i]);
			INTER_CLAMP(b_half_2[i] = b0[i] + third * timestep * b_grad_1[i]);
		}

		// Round 2, 1/3 step
		// input is only non-zero on the first round for simple upsampling
		input_L = 0.0;
		input_R = 0.0;

		#ifdef RESAMPLED_IO
		processInput(input_L, input_R, t_amp_out_L, t_amp_out_R);
		#endif
		
		modelIteration( a_half_2, a_grad_2, b_half_2, b_grad_2, input_L, input_R, t_amp_out_L, t_amp_out_R);

		#ifdef RESAMPLED_IO
		processOutput(t_amp_out_L, t_amp_out_R);
		#endif

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_3[i] = a0[i] + timestep * (-third * a_grad_1[i] + a_grad_2[i]));
			INTER_CLAMP(b_half_3[i] = b0[i] + timestep * (-third * b_grad_1[i] + b_grad_2[i]));
		}

		// Round 3, 2/3 step
		#ifdef RESAMPLED_IO
		processInput(input_L, input_R, t_amp_out_L, t_amp_out_R);
		#endif

		modelIteration( a_half_3, a_grad_3, b_half_3, b_grad_3, input_L, input_R, t_amp_out_L, t_amp_out_R);

		#ifdef RESAMPLED_IO
		processOutput(t_amp_out_L, t_amp_out_R);
		#endif

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			INTER_CLAMP(a_half_4[i] = a0[i] + timestep * (a_grad_1[i] - a_grad_2[i] + a_grad_3[i]));
			INTER_CLAMP(b_half_4[i] = b0[i] + timestep * (b_grad_1[i] - b_grad_2[i] + b_grad_3[i]));
		}

		// Round 4, whole step
		#ifdef RESAMPLED_IO
		processInput(input_L, input_R, t_amp_out_L, t_amp_out_R);
		#endif

		modelIteration( a_half_4, a_grad_4, b_half_4, b_grad_4, input_L, input_R, t_amp_out_L, t_amp_out_R);

		#ifdef RESAMPLED_IO
		processOutput(t_amp_out_L, t_amp_out_R);
		#endif

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			//final result
			//clamping isn't a part of RK4, but it's more convenient to do it here than elsewhere
			a1[i] = FINAL_CLAMP(a0[i] + timestep * (a_grad_1[i] + 3.0f * a_grad_2[i] + 3.0f * a_grad_3[i] + a_grad_4[i]) / 8.0f);
			b1[i] = FINAL_CLAMP(b0[i] + timestep * (b_grad_1[i] + 3.0f * b_grad_2[i] + 3.0f * b_grad_3[i] + b_grad_4[i]) / 8.0f);
		}

		// clamp to prevent blowups, but with a large range to avoid clipping in general
		#ifndef RESAMPLED_IO
		amp_out_L = math::clamp(t_amp_out_L,-100.0f,100.0f);
		amp_out_R = math::clamp(t_amp_out_R,-100.0f,100.0f);
		#else
		StereoSample out = getResampledOutput();
		amp_out_L = math::clamp(out.x, -100.0f, 100.0f);
		amp_out_R = math::clamp(out.y, -100.0f, 100.0f);
		#endif
	}

	float sum(float_4 x) {
		return x[0] + x[1] + x[2] + x[3];
	}

	void stepWaveEquation(
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
		
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

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
			delta_a[i] = (summed_probe_input + b + damping * laplacian_a[i] - decay * a);
			delta_b[i] = (laplacian_a[i] - decay * b);
		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void stepSquidAxon(
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {

		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		for (int i = 0; i < CHANNEL_SIZE; i++) {
			float_4 probe_in_L = t_input_L * input_probe_L_window[i];
			float_4 probe_in_R = t_input_R * input_probe_R_window[i];

			float_4 a = simd::clamp(a0[i],-1.0f,1.0f);
			float_4 b = simd::clamp(b0[i],-1.0f,1.0f);

			probe_out_L += a * output_probe_L_window[i];
			probe_out_R += a * output_probe_R_window[i];

			// Squid axon
			const float k1 = 1.0;
			const float k2 = 0.0;
			const float k3 = 1.0;
			float k4 = 0.995+0.1*decay;
			const float epsilon = 0.1;
			//#define delta 0.0
			float delta = 0.5*damping;
			const float ak0 = -0.1;
			const float ak1 = 2.0;

			float_4 summed_probe_input_a = 
					(additive_mode_L ? probe_in_L : (a * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (a * probe_in_R));
			float_4 summed_probe_input_b = 
					(additive_mode_L ? probe_in_L : (b * probe_in_L)) + 
					(additive_mode_R ? probe_in_R : (b * probe_in_R));
			delta_a[i] = summed_probe_input_a + k1*a - k2*a*a - k4*a*a*a - b + (1.0 + delta) * laplacian_a[i];
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
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
				  			
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

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
			delta_a[i] =  (-summed_probe_input_a + - laplacian_b[i] - decay * a + damping * laplacian_a[i]);
			delta_b[i] = ( summed_probe_input_b + laplacian_a[i] - decay * b + damping * laplacian_b[i]);
		}

		t_amp_out_L = sum(probe_out_L);
		t_amp_out_R = sum(probe_out_R);
	}

	void stepRK4Advection(		
		const std::vector<float_4> &a0, const std::vector<float_4> &b0, 
		const std::vector<float_4> &laplacian_a, const std::vector<float_4> &laplacian_b,
		const std::vector<float_4> &gradient_a, const std::vector<float_4> &gradient_b,
		std::vector<float_4> &delta_a, std::vector<float_4> &delta_b,
		const float &t_input_L, const float &t_input_R,
		float &t_amp_out_L, float &t_amp_out_R) {
			
		float_4 probe_out_L = float_4(0.0);
		float_4 probe_out_R = float_4(0.0);

		float_4 probe_in_R = simd::rescale(t_input_R,-10.0,10.0,-1.0,1.0);
		float pos_to_mod_offset = 2.0*(pos_in_R - (MAX_POSITION / 2.0)) / MAX_POSITION;
		probe_in_R = sig_in_R * (probe_in_R + pos_to_mod_offset);

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
			delta_a[i] = smoothclamp((summed_probe_input_a + damping * laplacian_a[i] - decay * a)  // input
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
	void generateProbeWindow(std::vector<float_4> &w, bool isDirty, float pos, float sigma, bool deriv) {
		if (isDirty || dirty_init) {
			float_4 w_sum = float_4(0.0);
			for (int i = 0; i < CHANNEL_SIZE; i++) {
				float_4 f_i = float_4(4.0*i, 4.0*i+1.0, 4.0*i+2.0, 4.0*i+3.0);
				if (deriv) {
					w[i] = approxGaussianDeriv(pos, f_i, sigma);
					w_sum += simd::abs(w[i]);
				} else {
					w[i] = approxGaussian(pos, f_i, sigma);
					w_sum += w[i];
				}
			}
			float w_norm = sum(w_sum);
			if (deriv) {
				w_norm *= 0.5;
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
		generateProbeWindow(input_probe_L_window, input_probe_L_dirty, this->pos_in_L, this->sig_in_L, differential_mode_L);
		generateProbeWindow(input_probe_R_window, input_probe_R_dirty, this->pos_in_R, this->sig_in_R, differential_mode_R);
		generateProbeWindow(output_probe_L_window, output_probe_L_dirty, this->pos_out_L, this->sig_out_L, false);
		generateProbeWindow(output_probe_R_window, output_probe_R_dirty, this->pos_out_R, this->sig_out_R, false);
	}

	void toggleAdditiveModeL() {
		additive_mode_L = !additive_mode_L;
	}

	void toggleAdditiveModeR() {
		additive_mode_R = !additive_mode_R;
	}

	void toggleDifferentialModeL() {
		differential_mode_L = !differential_mode_L;
		generateProbeWindow(input_probe_L_window, true, this->pos_in_L, this->sig_in_L, differential_mode_L);
	}

	void toggleDifferentialModeR() {
		differential_mode_R = !differential_mode_R;
		generateProbeWindow(input_probe_R_window, true, this->pos_in_R, this->sig_in_R, differential_mode_R);
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
				this->modelPointer = &WaveChannel::stepSquidAxon;
				break;
			case SQUID_AXON:
				this->model = Model::SCHRODINGER;
				this->modelPointer = &WaveChannel::stepSchrodinger;
				break;
			case SCHRODINGER:
				this->model = Model::RK4_ADVECTION;
				this->modelPointer = &WaveChannel::stepRK4Advection;
				break;
			case RK4_ADVECTION:
				this->model = Model::WAVE_EQUATION;
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

	float getAmpOutL() {
		return amp_out_L;
	}

	float getAmpOutR() {
		return amp_out_R;
	}

	// Update the ping-pong buffers
	void update() {
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
		DIFFERENTIAL_BUTTON_L_PARAM,
		MULTIPLICATIVE_BUTTON_L_PARAM,
		DIFFERENTIAL_BUTTON_R_PARAM,
		MULTIPLICATIVE_BUTTON_R_PARAM,
		POSITION_IN_L_PARAM,
		POSITION_IN_R_PARAM,
		POSITION_IN_L_CV_PARAM,
		POSITION_IN_R_CV_PARAM,
		PROBE_SIGMA_IN_L_PARAM,
		PROBE_SIGMA_IN_R_PARAM,
		PROBE_SIGMA_IN_L_CV_PARAM,
		PROBE_SIGMA_IN_R_CV_PARAM,
		POSITION_OUT_L_PARAM,
		POSITION_OUT_R_PARAM,
		POSITION_OUT_L_CV_PARAM,
		POSITION_OUT_R_CV_PARAM,
		PROBE_SIGMA_OUT_L_PARAM,
		PROBE_SIGMA_OUT_R_PARAM,
		PROBE_SIGMA_OUT_L_CV_PARAM,
		PROBE_SIGMA_OUT_R_CV_PARAM,
		DAMPING_CV_PARAM,
		TIMESTEP_CV_PARAM,
		DECAY_CV_PARAM,
		FEEDBACK_CV_PARAM,
		LOW_CUT_CV_PARAM,
		INPUT_GAIN_L_CV_PARAM,
		INPUT_GAIN_R_CV_PARAM,
		DRY_CV_PARAM,
		WET_CV_PARAM,
		DAMPING_PARAM,
		TIMESTEP_PARAM,
		DECAY_PARAM,
		FEEDBACK_PARAM,
		LOW_CUT_PARAM,
		INPUT_GAIN_L_PARAM,
		INPUT_GAIN_R_PARAM,
		DRY_PARAM,
		WET_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		PROBE_IN_L_INPUT,
		PROBE_IN_R_INPUT,
		PROBE_POSITION_IN_L_INPUT,
		PROBE_POSITION_IN_R_INPUT,
		PROBE_SIGMA_IN_L_INPUT,
		PROBE_SIGMA_IN_R_INPUT,
		PROBE_POSITION_OUT_L_INPUT,
		PROBE_POSITION_OUT_R_INPUT,
		PROBE_SIGMA_OUT_L_INPUT,
		PROBE_SIGMA_OUT_R_INPUT,
		DAMPING_INPUT,
		TIMESTEP_INPUT,
		DECAY_INPUT,
		FEEDBACK_INPUT,
		LOW_CUT_INPUT,
		INPUT_GAIN_L_INPUT,
		INPUT_GAIN_R_INPUT,
		DRY_INPUT,
		WET_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		PROBE_OUT_L_OUTPUT,
		PROBE_OUT_R_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		EOC_LIGHT,
		INTEGRAL_L_LIGHT,
		ADDITIVE_L_LIGHT,
		DIFFERENTIAL_L_LIGHT,
		MULTIPLICATIVE_L_LIGHT,
		INTEGRAL_R_LIGHT,
		ADDITIVE_R_LIGHT,
		DIFFERENTIAL_R_LIGHT,
		MULTIPLICATIVE_R_LIGHT,
		POS_MODE_LIGHT,
		MOD_MODE_LIGHT,
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
	CVParamInput<TIMESTEP_PARAM, TIMESTEP_INPUT,  TIMESTEP_CV_PARAM> timestep_param;
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

	#define TIMESTEP_SHIFT 3.191
	#define TIMESTEP_MAX 0.4
	#define TIMESTEP_KNOB_MIN -8.0
	#define TIMESTEP_KNOB_MAX 8.0
	#define TIMESTEP_DEF 0.0
	#define TIMESTEP_SLEW_RATE 0.02f
	#define TIMESTEP_POST_SCALE 1.5

	#define FEEDBACK_MIN -8.0
	#define FEEDBACK_MAX 8.0
	#define FEEDBACK_DEF 0.0

	
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
		feedback_param.config(this, FEEDBACK_MIN, FEEDBACK_MAX, FEEDBACK_DEF, "feedback", "Feedback");
		low_cut_param.config(this, LOW_CUT_MIN, LOW_CUT_MAX, LOW_CUT_DEF, "low_cut", "Low Cut");
		input_gain_L_param.configExp(this, 0.0, 8.0, 6.0, "input_gain_L", "Input Gain L");
		input_gain_R_param.configExp(this, 0.0, 8.0, 6.0, "input_gain_R", "Input Gain R");
		dry_param.configExp(this, 0.0, 8.0, 0.0, "dry", "Dry Gain");
		wet_param.configExp(this, 0.0, 8.0, 6.0, "wet", "Wet Gain");
		lightDivider.setDivision(16);
	}

	bool anyOutputsConnected() {
		return outputs[PROBE_OUT_L_OUTPUT].isConnected() || outputs[PROBE_OUT_R_OUTPUT].isConnected();
	}

	void setNextModel() {
		waveChannel.setNextModel();
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

				float diff_light_l = waveChannel.differential_mode_L ? 1.0 : 0.0;
				float int_light_l = waveChannel.differential_mode_L ? 0.0 : 1.0;
				float add_light_l = waveChannel.additive_mode_L ? 1.0 : 0.0;
				float mult_light_l = waveChannel.additive_mode_L ? 0.0 : 1.0;

				float add_light_r;
				float mult_light_r;
				float diff_light_r;
				float int_light_r;
				if (disable_R_diff_add_lights) {
					add_light_r = 0.0;
					mult_light_r = 0.0;
					diff_light_r = 0.0;
					int_light_r = 0.0;
				} else {
					add_light_r = waveChannel.additive_mode_R ? 1.0 : 0.0;
					mult_light_r = waveChannel.additive_mode_R ? 0.0 : 1.0;
					diff_light_r = waveChannel.differential_mode_R ? 1.0 : 0.0;
					int_light_r = waveChannel.differential_mode_R ? 0.0 : 1.0;
				}

				lights[POS_MODE_LIGHT].setBrightness(pos_light);
				lights[MOD_MODE_LIGHT].setBrightness(mod_light);

				lights[DIFFERENTIAL_L_LIGHT].setBrightness(diff_light_l);
				lights[INTEGRAL_L_LIGHT].setBrightness(int_light_l);
				lights[DIFFERENTIAL_R_LIGHT].setBrightness(diff_light_r);
				lights[INTEGRAL_R_LIGHT].setBrightness(int_light_r);

				lights[ADDITIVE_L_LIGHT].setBrightness(add_light_l);
				lights[MULTIPLICATIVE_L_LIGHT].setBrightness(mult_light_l);
				lights[ADDITIVE_R_LIGHT].setBrightness(add_light_r);
				lights[MULTIPLICATIVE_R_LIGHT].setBrightness(mult_light_r);

				lights[POS_MODE_LIGHT].setBrightness(pos_light);
				lights[MOD_MODE_LIGHT].setBrightness(mod_light);
		}
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

	json_t* dataToJson() override {
		json_t* rootJ = json_object();
		pos_in_L_param.dataToJson(rootJ);
		pos_in_R_param.dataToJson(rootJ);
		pos_out_L_param.dataToJson(rootJ);
		pos_out_R_param.dataToJson(rootJ);
		booleanToJson(rootJ, waveChannel.differential_mode_L, "differential_mode_L");
		booleanToJson(rootJ, waveChannel.differential_mode_R, "differential_mode_R");
		booleanToJson(rootJ, waveChannel.additive_mode_L, "additive_mode_L");
		booleanToJson(rootJ, waveChannel.additive_mode_R, "additive_mode_R");
		modelToJson(rootJ, waveChannel.model, "model");
		return rootJ;
	}

	void dataFromJson(json_t* rootJ) override {
		pos_in_L_param.dataFromJson(rootJ);
		pos_in_R_param.dataFromJson(rootJ);
		pos_out_L_param.dataFromJson(rootJ);
		pos_out_R_param.dataFromJson(rootJ);
		booleanFromJson(rootJ, waveChannel.differential_mode_L, "differential_mode_L");
		booleanFromJson(rootJ, waveChannel.differential_mode_R, "differential_mode_R");
		booleanFromJson(rootJ, waveChannel.additive_mode_L, "additive_mode_L");
		booleanFromJson(rootJ, waveChannel.additive_mode_R, "additive_mode_R");
		modelFromJson(rootJ, waveChannel.model, "model");
	}
};


struct WaterTableWidget : ModuleWidget {
	WaterTableWidget(WaterTable* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/WaterTable.svg")));

		{
			WaterTableModeButton<WaterTable>* modelButton = createParamCentered<WaterTableModeButton<WaterTable>>(mm2px(Vec(74.5, 78.5)), module, WaterTable::MODEL_BUTTON_PARAM);
			modelButton->module = module;
			addParam(modelButton);
		}

		{
			WaterTableAdditiveModeLToggle<WaterTable>* modelButton = createParamCentered<WaterTableAdditiveModeLToggle<WaterTable>>(mm2px(Vec(23.917, 71.29)), module, WaterTable::MULTIPLICATIVE_BUTTON_L_PARAM);
			modelButton->module = module;
			addParam(modelButton);
		}

		{
			WaterTableAdditiveModeRToggle<WaterTable>* modelButton = createParamCentered<WaterTableAdditiveModeRToggle<WaterTable>>(mm2px(Vec(57.064, 71.29)), module, WaterTable::MULTIPLICATIVE_BUTTON_R_PARAM);
			modelButton->module = module;
			addParam(modelButton);
		}

		{
			WaterTableDifferentialModeLToggle<WaterTable>* modelButton = createParamCentered<WaterTableDifferentialModeLToggle<WaterTable>>(mm2px(Vec(6.442, 71.29)), module, WaterTable::DIFFERENTIAL_BUTTON_L_PARAM);
			modelButton->module = module;
			addParam(modelButton);
		}

		{
			WaterTableDifferentialModeRToggle<WaterTable>* modelButton = createParamCentered<WaterTableDifferentialModeRToggle<WaterTable>>(mm2px(Vec(39.589, 71.29)), module, WaterTable::DIFFERENTIAL_BUTTON_R_PARAM);
			modelButton->module = module;
			addParam(modelButton);
		}

		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(48.056, 101.879)), module,  WaterTable::POSITION_OUT_R_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(15.162, 101.948)), module,  WaterTable::POSITION_OUT_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(15.197, 25.839)), module, WaterTable::POSITION_IN_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(48.183, 25.839)), module, WaterTable::POSITION_IN_R_PARAM));

		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.197, 41.881)), module, WaterTable::PROBE_SIGMA_IN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(48.183, 41.881)), module, WaterTable::PROBE_SIGMA_IN_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.244, 53.504)), module, WaterTable::INPUT_GAIN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(48.248, 53.597)), module, WaterTable::INPUT_GAIN_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(99.469, 88.337)), module,   WaterTable::DRY_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(110.112, 88.337)), module,  WaterTable::WET_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(69.13, 117.289)), module,   WaterTable::DAMPING_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(79.772, 117.325)), module,  WaterTable::TIMESTEP_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(90.414, 117.325)), module,  WaterTable::DECAY_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(101.057, 117.325)), module, WaterTable::FEEDBACK_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(111.699, 117.325)), module, WaterTable::LOW_CUT_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(48.056, 117.922)), module,  WaterTable::PROBE_SIGMA_OUT_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.162, 117.991)), module,  WaterTable::PROBE_SIGMA_OUT_L_PARAM));

		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.442, 34.594)), module,  WaterTable::POSITION_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(39.428, 34.594)), module, WaterTable::POSITION_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.442, 45.726)), module,  WaterTable::PROBE_SIGMA_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(39.428, 45.726)), module, WaterTable::PROBE_SIGMA_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.442, 59.945)), module,  WaterTable::INPUT_GAIN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(39.446, 60.039)), module, WaterTable::INPUT_GAIN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(99.469, 71.989)), module,  WaterTable::DRY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(110.112, 71.989)), module, WaterTable::WET_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(68.919, 100.941)), module,  WaterTable::DAMPING_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(79.75,  100.977)), module,   WaterTable::TIMESTEP_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(90.414, 100.977)), module,  WaterTable::DECAY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(101.057, 100.977)), module, WaterTable::FEEDBACK_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(111.699, 100.977)), module, WaterTable::LOW_CUT_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(39.3, 110.635)), module,    WaterTable::POSITION_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.406, 110.703)), module,   WaterTable::POSITION_OUT_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(39.3, 121.766)), module,    WaterTable::PROBE_SIGMA_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.406, 121.835)), module,   WaterTable::PROBE_SIGMA_OUT_L_CV_PARAM));
		
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.197, 8.433)), module, WaterTable::PROBE_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(48.183, 8.433)), module, WaterTable::PROBE_IN_R_INPUT));

		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.953, 34.594)), module, WaterTable::PROBE_POSITION_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(56.939, 34.594)), module, WaterTable::PROBE_POSITION_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.953, 45.726)), module, WaterTable::PROBE_SIGMA_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(56.939, 45.726)), module, WaterTable::PROBE_SIGMA_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.917, 59.945)), module, WaterTable::INPUT_GAIN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(56.921, 60.039)), module, WaterTable::INPUT_GAIN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(99.469,  80.0)), module, WaterTable::DRY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(110.112, 80.0)), module, WaterTable::WET_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(68.919,  109.0)), module, WaterTable::DAMPING_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(79.75,   109.0)), module, WaterTable::TIMESTEP_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(90.414,  109.0)), module, WaterTable::DECAY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(101.057, 109.0)), module, WaterTable::FEEDBACK_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(111.699, 109.0)), module, WaterTable::LOW_CUT_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(56.811, 110.635)), module,  WaterTable::PROBE_POSITION_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.917, 110.703)), module,  WaterTable::PROBE_POSITION_OUT_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(56.811, 121.766)), module,  WaterTable::PROBE_SIGMA_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.917, 121.835)), module,  WaterTable::PROBE_SIGMA_OUT_L_INPUT));

		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.162, 84.976)), module, WaterTable::PROBE_OUT_L_OUTPUT));
		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(48.056, 85.044)), module, WaterTable::PROBE_OUT_R_OUTPUT));

		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(115.574, 4.806)), module, WaterTable::EOC_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(10.969, 66.884)), module, WaterTable::INTEGRAL_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(19.519, 66.884)), module, WaterTable::ADDITIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(10.969, 75.573)), module, WaterTable::DIFFERENTIAL_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(19.519, 75.573)), module, WaterTable::MULTIPLICATIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(43.973, 66.884)), module, WaterTable::INTEGRAL_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(52.523, 66.884)), module, WaterTable::ADDITIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(43.973, 75.573)), module, WaterTable::DIFFERENTIAL_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(52.523, 75.573)), module, WaterTable::MULTIPLICATIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(48.248, 68.917)), module, WaterTable::POS_MODE_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(48.248, 73.663)), module, WaterTable::MOD_MODE_LIGHT));

		{
			WaterTableDisplay<WaterTable, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>* display = new WaterTableDisplay<WaterTable, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>();
			display->module = module;
			display->box.pos = mm2px(Vec(64.349, 9.249));
			display->box.size = mm2px(Vec(52.213, 57.568));
			display->setBBox();
			addChild(display);
		}

	}
};


Model* modelWaterTable = createModel<WaterTable, WaterTableWidget>("FreeSurface-WaterTable");
