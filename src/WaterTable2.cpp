#include "plugin.hpp"
#include "OpCache.hpp"
//#include "Profiler.hpp"
#include <cmath>
#include <cstring>
#include "Integrator.h"

using simd::float_4;
using simd::int32_4;

struct WaveChannel2 {
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
    float anisotropy = 0.0;

	// ping pong buffer setup
	bool pong = false;

	ProbeType input_probe_type_L = ProbeType::INTEGRAL;
	ProbeType input_probe_type_R = ProbeType::INTEGRAL;
	ProbeType output_probe_type_L = ProbeType::INTEGRAL;
	ProbeType output_probe_type_R = ProbeType::INTEGRAL;
	bool additive_mode_L = true;
	bool additive_mode_R = true;

	// weights for the input and output probes, generated whenever the respective probe settings change
	std::vector<float_4> input_probe_L_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> input_probe_R_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> output_probe_L_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());
	std::vector<float_4> output_probe_R_window = std::vector<float_4>(CHANNEL_SIZE, float_4::zero());

    // kernel weights
    std::vector<float_4> kernel_weight_N = std::vector<float_4>(CHANNEL_SIZE, float_4(1.));
    std::vector<float_4> kernel_weight_E = std::vector<float_4>(CHANNEL_SIZE, float_4(1.));
    std::vector<float_4> kernel_weight_S = std::vector<float_4>(CHANNEL_SIZE, float_4(1.));
    std::vector<float_4> kernel_weight_W = std::vector<float_4>(CHANNEL_SIZE, float_4(1.));
    std::vector<float_4> kernel_weight_C = std::vector<float_4>(CHANNEL_SIZE, float_4(-4.));

	WaveChannel2() {
		model = Model::WAVE_EQUATION;
	}

    // TODO: will decide later on if I want to do 1D or 2D buffers throughout.
    //  in the latter case this function should not be needed
    // This works with unpacked float indices and is used for computing positions
    // rather than indexing into a buffer
    inline void indexToPosFloats(unsigned int index, unsigned int &x, unsigned int &y) {
        x = index & CHANNEL_MASK_X_FLOATS;
        y = index / CHANNEL_SIZE_X_FLOATS;
    }

    inline void indexToPos(unsigned int index, unsigned int &x, unsigned int &y) {
        x = index & CHANNEL_MASK_X;
        y = index / CHANNEL_SIZE_X;
    }

    inline unsigned int posToIndex(unsigned int x, unsigned int y) {
        return x + y * CHANNEL_SIZE_X;
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
    // TODO: I don't think this is still valid if used for 2D coords?
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

	/** computes a gaussian using the differences of the (approximate) error function
	 *  this is a much better approach than simply sampling a gaussian because we can
	 *  almost eliminate aliasing, even at small kernel sizes
	 *  TODO: as much as I like all the little math tricks here, this stuff is overly slow
	 */

    // TODO: this will need to be substantially changed to deal with 2D
	inline float_4 approxGaussian(float_4 x, float_4 sig) {
		// float_4 x_s = wrappedSignedDistance(x, mean);
        float_4 x_s = x;
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
	 *  TODO: this will need to be substantially changed to deal with 2D
	 */
	inline float_4 approxGaussianDeriv(float_4 x, float_4 sig) {
		float_4 x_s = x;
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
	 *  TODO: above statement is not strictly speaking true,
	 *   could be done using a table for the Si(x) function, which
	 *   may also be faster than calling simd::sin below.
	 *  TODO: this will need to be substantially changed to deal with 2D
	 */
	inline float_4 sinc(float_4 x, float_4 sig) {
		float_4 x_s = x;
		float_4 x_s2 = x_s / sig;
		float_4 snc = simd::sin(x_s2)/x_s2;
		float_4 almost_zero = simd::abs(x_s2) < 1.0e-6f;
		return simd::ifelse(almost_zero, 1.0, snc);
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

    bool kernel_weights_dirty = true;
    bool kernel_weights_dirty_init = true;

	void setDirtyProbe(bool& dirty_flag, const float& pos_prev, const float& sig_prev, const float& pos_next, const float& sig_next) {
		dirty_flag = (pos_prev != pos_next || sig_prev != sig_next);
	}

    void setDirtyKernelWeights(bool& dirty_flag, const float& anisotropy_prev, const float& anisotropy_next) {
        dirty_flag = (anisotropy_prev != anisotropy_next);
    }

	// Generate and normalize probe window buffers
	void generateProbeWindow(std::vector<float_4> &w, bool isDirty, float pos, float sigma, ProbeType probeType) {
		if (isDirty || dirty_init) {
			float_4 w_sum = float_4(0.0);
            // TODO: until I make probe position 2D throughout, don't bother with smooth probe position transitions
            unsigned int posu = static_cast<unsigned int>(pos);
            unsigned int posx;
            unsigned int posy;
            indexToPosFloats(posu, posx, posy);
            float_4 p_x = float_4(static_cast<float>(posx));
            float_4 p_y = float_4(static_cast<float>(posy));

			for (unsigned int i = 0; i < CHANNEL_SIZE_X; i++) {
                for (unsigned int j = 0; j < CHANNEL_SIZE_Y; j++) {
                    float_4 f_x = float_4(4.0 * i, 4.0 * i + 1.0, 4.0 * i + 2.0, 4.0 * i + 3.0) - p_x;
                    float_4 f_y = float_4(1.0 * j) - p_y;
                    float_4 f_i = sqrt(f_x*f_x + f_y*f_y);
                    unsigned int idx = posToIndex(i, j);
                    switch (probeType) {
                        case ProbeType::INTEGRAL:
                            w[idx] = approxGaussian(f_i, sigma);
                            w_sum += w[idx];
                            break;
                        case ProbeType::DIFFERENTIAL:
                            w[idx] = approxGaussianDeriv(f_i, sigma);
                            w_sum += simd::abs(w[idx]);
                            break;
                        case ProbeType::SINC:
                            w[idx] = sinc(f_i, sigma);
                            w_sum += simd::abs(w[idx]);
                            break;
                    }
                }
			}
			float w_norm = sum(w_sum);
			if (probeType == ProbeType::SINC) {
				w_norm *= 0.5; //not the correct factor probably, but close
			}
			for (int i = 0; i < CHANNEL_SIZE; i++) {
				w[i] /= w_norm;
			}

            // WARNING! This will generate a HUGE amount of data in the log file
            // when knobs are turned, or CV input is connected to the knob position/sigma
            //#define DEBUG_PROBE_PRINT
			#ifdef DEBUG_PROBE_PRINT
				std::string debug_string;
				for (auto f : w) {
					debug_string += std::to_string(f[0]) + " " + std::to_string(f[1]) + " " + std::to_string(f[2]) + " " + std::to_string(f[3]) + " ";
				}
				debug_string = "probe_generated: " + debug_string;
				//INFO(debug_string.c_str());
                std::cout << "posu " << posu << std::endl;
                std::cout << "posx " << posx << std::endl;
                std::cout << "posy " << posy << std::endl;
                std::cout << debug_string << std::endl;
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

    void setKernelSettings(float anisotropy) {
        setDirtyKernelWeights(kernel_weights_dirty, this->anisotropy, anisotropy);
        this->anisotropy = anisotropy;
        generateKernelWeights(anisotropy, kernel_weights_dirty);
    }

    /*
     * Design considerations for kernel weight modification
     * It seems sensible to store separate kernel weights in separate arrays of float_4s
     * for each cardinal direction, in order to match the structure of the laplacian calculation.
     * The center (vertex) weight for the laplacian stencil is simply the negation of the sum of edge weights.
     * This means we could just store the edge weights and not the vertex weight, but that also entails
     * computing the vertex weight every time we need to compute the laplacian. So, it seems sensible to compute
     * it ahead of time, meaning we'll also compute a vertex weight array. This computation is, on the whole,
     * a bit simpler than what we're doing to compute antialiased probe weights, but it will still be relatively
     * expensive, meaning that we should use a similar approach to the above to change the weights only when needed.
     *
     * For an initial test run of this idea, we'll simply provide a control for the anisotropy of the kernel.
     */

    // let's say for now that anisotropy = 1 means maximum horizontal / vertical ratio,
    // and anistropy = -1 means maximum vertical / horizontal ratio
    // For stability reasons, we'll probably want to fix the larger of the two values to 1,
    // although through testing we can find if larger values are feasible
    void generateKernelWeights(float anisotropy, bool isDirty) {
        if (isDirty || kernel_weights_dirty_init) {
            float h_weight = 1.;
            float v_weight = 1.;
            if (anisotropy < 0.) {
                h_weight = 1./(1. - anisotropy * 4.);
            } else {
                v_weight = 1./(1. + anisotropy * 4.);
            }
            float c_weight = -(2.*h_weight + 2.*v_weight);
            for (int i = 0; i < CHANNEL_SIZE_X; i++) {
                for (int j = 0; j < CHANNEL_SIZE_Y; j++) {
                    unsigned int idx = posToIndex(i, j);
                    kernel_weight_N[idx] = float_4(v_weight);
                    kernel_weight_S[idx] = float_4(v_weight);
                    kernel_weight_E[idx] = float_4(h_weight);
                    kernel_weight_W[idx] = float_4(h_weight);
                    kernel_weight_C[idx] = float_4(c_weight);
                }
            }
        }
        kernel_weights_dirty_init = false;
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
	}

	const char* getModelString() {
		const char* text;
		switch(model) {
			case WaveChannel2::Model::WAVE_EQUATION:
				text = "WAVE_EQUATION";
				break;
			case WaveChannel2::Model::SCHRODINGER:
				text = "SCHRODINGER";
				break;
			case WaveChannel2::Model::RK4_ADVECTION:
				text = "RUNGE_KUTTA_RK4";
				break;
			case WaveChannel2::Model::SQUID_AXON:
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
			case WaveChannel2::Model::RK4_ADVECTION:
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

    ModelIter_Data<2, CHANNEL_SIZE> dataPing;
    ModelIter_Data<2, CHANNEL_SIZE> dataPong = dataPing.create_swapped_copy();
    ModelParams<CHANNEL_SIZE> modelParams;

	// Update the ping-pong buffers
	void update() {
		setClipRange();
		//setModelPointer();
        ModelIter_Data<2, CHANNEL_SIZE>* data;
        if (pong) {
            data = &dataPing;
        } else {
            data = &dataPong;
        }

        switch(this->model) {
            case SQUID_AXON:
                RK4_iter_3_8s<2, CHANNEL_SIZE, stepSquidAxon<2, CHANNEL_SIZE>>(*data, modelParams);
                break;
            case SCHRODINGER:
                RK4_iter_3_8s<2, CHANNEL_SIZE, stepSchrodinger<2, CHANNEL_SIZE>>(*data, modelParams);
                break;
            case RK4_ADVECTION:
                RK4_iter_3_8s<2, CHANNEL_SIZE, stepRK4Advection<2, CHANNEL_SIZE>>(*data, modelParams);
                break;
            case WAVE_EQUATION:
                RK4_iter_3_8s<2, CHANNEL_SIZE, stepWaveEquation<2, CHANNEL_SIZE>>(*data, modelParams);
                break;
        }

		pong = !pong;
	}
};

struct WaterTable2 : Module {
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
        ANISOTROPY_PARAM,
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
        ANISOTROPY_INPUT,
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

	WaveChannel2 waveChannel;
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

    CVParamInput<ANISOTROPY_PARAM, ANISOTROPY_INPUT, DUMMY_CV> anisotropy_param;

	
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

    // TODO: this should be specified in dB
	#define MIN_GAIN 0.0
	#define MAX_GAIN 8.0
	#define DEF_GAIN 6.0

	
	WaterTable2() : timestepSlewLimiter(TIMESTEP_SLEW_RATE) {
		
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
        anisotropy_param.config(this, -1.0, 1.0, 0.0, "anisotropy", "Anisotropy");
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

	void setLightPatternProbeType(WaveChannel2::ProbeType probeType, float &integralLight, float &differentialLight, float &sincLight, bool override) {
		switch(probeType) {
			case WaveChannel2::ProbeType::DIFFERENTIAL:
				differentialLight = 1.0;
				integralLight = 0.0;
				sincLight = 0.0;
				break;
			case WaveChannel2::ProbeType::INTEGRAL:
				differentialLight = 0.0;
				integralLight = 1.0;
				sincLight = 0.0;
				break;
			case WaveChannel2::ProbeType::SINC:
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
		waveChannel.input_probe_type_L = WaveChannel2::ProbeType::INTEGRAL;
		waveChannel.input_probe_type_R = WaveChannel2::ProbeType::INTEGRAL;
		waveChannel.output_probe_type_L = WaveChannel2::ProbeType::INTEGRAL;
		waveChannel.output_probe_type_R = WaveChannel2::ProbeType::INTEGRAL;
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
        float anisotropy = anisotropy_param.getValue();

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
            waveChannel.setKernelSettings(anisotropy);
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
					case WaveChannel2::Model::RK4_ADVECTION:
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
		waveChannel.oversampling_mode = static_cast<WaveChannel2::OversamplingMode>(mode);
	}

	int getClipRangeMode() {
		return static_cast<int>(waveChannel.clip_range_mode);
	}

	void setClipRangeMode(int mode) {
		waveChannel.clip_range_mode = static_cast<WaveChannel2::ClipRange>(mode);
	}

	void booleanFromJson(json_t* rootJ, bool &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = json_boolean_value(j_val);
	}

	void booleanToJson(json_t* rootJ, bool &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_boolean(val));
	}

	void modelFromJson(json_t* rootJ, WaveChannel2::Model &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel2::Model>(json_integer_value(j_val));
	}

	void modelToJson(json_t* rootJ, WaveChannel2::Model &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void probeFromJson(json_t* rootJ, WaveChannel2::ProbeType &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel2::ProbeType>(json_integer_value(j_val));
	}

	void probeToJson(json_t* rootJ, WaveChannel2::ProbeType &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void oversamplingModeFromJson(json_t* rootJ, WaveChannel2::OversamplingMode &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel2::OversamplingMode>(json_integer_value(j_val));
	}

	void oversamplingModeToJson(json_t* rootJ, WaveChannel2::OversamplingMode &val, const char* json_label) {
		json_object_set_new(rootJ, json_label, json_integer(static_cast<int>(val)));
	}

	void clipRangeModeFromJson(json_t* rootJ, WaveChannel2::ClipRange &val, const char* json_label) {
		json_t* j_val = json_object_get(rootJ, json_label);
		if (j_val)
			val = static_cast<WaveChannel2::ClipRange>(json_integer_value(j_val));
	}

	void clipRangeModeToJson(json_t* rootJ, WaveChannel2::ClipRange &val, const char* json_label) {
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
		probeFromJson(rootJ,   waveChannel.input_probe_type_L, "input_probe_type_L");
		probeFromJson(rootJ,   waveChannel.input_probe_type_R, "input_probe_type_R");
		probeFromJson(rootJ,   waveChannel.output_probe_type_L, "output_probe_type_L");
		probeFromJson(rootJ,   waveChannel.output_probe_type_R, "output_probe_type_R");
		booleanFromJson(rootJ, waveChannel.additive_mode_L, "additive_mode_L");
		booleanFromJson(rootJ, waveChannel.additive_mode_R, "additive_mode_R");
		modelFromJson(rootJ,   waveChannel.model, "model");
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


struct WaterTable2Widget : ModuleWidget {
	WaterTable2Widget(WaterTable2* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/WaterTable.svg")));

		/*  lambdas below MUST BE pass by value or we segfault when the reference goes out of scope.
			this way of setting up the buttons is somewhat bizarre, but it does reduce boilerplate substantially.
		*/
		{
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(69.566, 83.327)), module, WaterTable2::MODEL_BUTTON_PARAM));
			FreeSurfaceLogoToggleDark<WaterTable2, 4>* button 
					= createParamCentered<FreeSurfaceLogoToggleDark<WaterTable2, 4>>(mm2px(Vec(69.566, 83.327)), module, WaterTable2::MODEL_BUTTON_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(18.94, 66.2)), module, WaterTable2::MULTIPLICATIVE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable2, 2>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 2>>(mm2px(Vec(18.94, 66.2)), module, WaterTable2::MULTIPLICATIVE_BUTTON_L_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(48.062, 66.2)), module, WaterTable2::MULTIPLICATIVE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable2, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 3>>(mm2px(Vec(48.062, 66.2)), module, WaterTable2::MULTIPLICATIVE_BUTTON_R_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(5.018, 66.198)), module, WaterTable2::INPUT_PROBE_TYPE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable2, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 3>>(mm2px(Vec(5.018, 66.198)), module, WaterTable2::INPUT_PROBE_TYPE_BUTTON_L_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(34.14, 66.198)), module, WaterTable2::INPUT_PROBE_TYPE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable2, 4>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 4>>(mm2px(Vec(34.14, 66.198)), module, WaterTable2::INPUT_PROBE_TYPE_BUTTON_R_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(4.991, 121.739)), module, WaterTable2::OUTPUT_PROBE_TYPE_BUTTON_L_PARAM));
			RoundToggleDark<WaterTable2, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 3>>(mm2px(Vec(4.991, 121.739)), module, WaterTable2::OUTPUT_PROBE_TYPE_BUTTON_L_PARAM);
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
			//addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(34.287, 121.739)), module, WaterTable2::OUTPUT_PROBE_TYPE_BUTTON_R_PARAM));
			RoundToggleDark<WaterTable2, 3>* button 
					= createParamCentered<RoundToggleDark<WaterTable2, 3>>(mm2px(Vec(34.287, 121.739)), module, WaterTable2::OUTPUT_PROBE_TYPE_BUTTON_R_PARAM);
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
			WaterTable2Display<WaterTable2, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>* display = new WaterTable2Display<WaterTable2, CHANNEL_SIZE, CHANNEL_SIZE_FLOATS>();
			display->module = module;
			display->box.pos = mm2px(Vec(59.822, 9.072));
			display->box.size = mm2px(Vec(60.444, 62.491));
			display->setBBox();
			addChild(display);
		}
		


		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(23.75, 19.313)), module, WaterTable2::POSITION_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(53.046, 19.313)), module, WaterTable2::POSITION_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 40.993)), module, WaterTable2::PROBE_SIGMA_IN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.717, 40.993)), module, WaterTable2::PROBE_SIGMA_IN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 53.047)), module, WaterTable2::INPUT_GAIN_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.676, 53.047)), module, WaterTable2::INPUT_GAIN_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(23.75, 89.313)), module, WaterTable2::POSITION_OUT_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(53.046, 89.313)), module, WaterTable2::POSITION_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(83.969, 104.575)), module, WaterTable2::DAMPING_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(94.324, 104.575)), module, WaterTable2::DECAY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(104.848, 104.575)), module, WaterTable2::FEEDBACK_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(115.382, 104.575)), module, WaterTable2::LOW_CUT_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(6.421, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_L_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(35.676, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_R_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(95.75, 93.843)), module, WaterTable2::DRY_CV_PARAM));
		addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(115.288, 93.843)), module, WaterTable2::WET_CV_PARAM));

		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(10.25, 24.063)), module, WaterTable2::POSITION_IN_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(39.546, 24.063)), module, WaterTable2::POSITION_IN_R_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(10.25, 94.063)), module, WaterTable2::POSITION_OUT_L_PARAM));
		addParam(createParamCentered<VektronixInfiniteBigKnob>(mm2px(Vec(39.546, 94.063)), module, WaterTable2::POSITION_OUT_R_PARAM));

		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(69.566, 109.631)), module, WaterTable2::TIMESTEP_PARAM));
		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(110.538, 81.989)), module, WaterTable2::WET_PARAM));
		addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(91.0, 82.107)), module, WaterTable2::DRY_PARAM));

        addParam(createParamCentered<VektronixTinyKnobDark>(mm2px(Vec(110.538, 70.0)), module, WaterTable2::ANISOTROPY_PARAM));
        addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(100, 70.0)), module, WaterTable2::ANISOTROPY_INPUT));

		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 40.64)), module, WaterTable2::PROBE_SIGMA_IN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 40.64)), module, WaterTable2::PROBE_SIGMA_IN_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_R_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(83.969, 112.384)), module, WaterTable2::DAMPING_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(94.324, 112.392)), module, WaterTable2::DECAY_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(104.848, 112.392)), module, WaterTable2::FEEDBACK_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(115.382, 112.392)), module, WaterTable2::LOW_CUT_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(15.222, 52.798)), module, WaterTable2::INPUT_GAIN_L_PARAM));
		addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(44.176, 52.798)), module, WaterTable2::INPUT_GAIN_R_PARAM));

		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.222, 7.244)), module, WaterTable2::PROBE_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(44.296, 7.244)), module, WaterTable2::PROBE_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 28.063)), module, WaterTable2::PROBE_POSITION_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 28.063)), module, WaterTable2::PROBE_POSITION_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 40.64)), module, WaterTable2::PROBE_SIGMA_IN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 40.64)), module, WaterTable2::PROBE_SIGMA_IN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 52.691)), module, WaterTable2::INPUT_GAIN_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 52.691)), module, WaterTable2::INPUT_GAIN_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(87.0, 93.843)), module, WaterTable2::DRY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(106.538, 93.843)), module, WaterTable2::WET_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 98.063)), module, WaterTable2::PROBE_POSITION_OUT_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 98.063)), module, WaterTable2::PROBE_POSITION_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(23.75, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_L_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(53.046, 110.41)), module, WaterTable2::PROBE_SIGMA_OUT_R_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(73.56, 120.947)), module, WaterTable2::TIMESTEP_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(83.969, 120.947)), module, WaterTable2::DAMPING_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(94.324, 120.947)), module, WaterTable2::DECAY_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(104.848, 120.947)), module, WaterTable2::FEEDBACK_INPUT));
		addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(115.382, 120.947)), module, WaterTable2::LOW_CUT_INPUT));

		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(15.222, 78.379)), module, WaterTable2::PROBE_OUT_L_OUTPUT));
		addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(44.296, 78.379)), module, WaterTable2::PROBE_OUT_R_OUTPUT));


		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(118.795, 4.673)), module, WaterTable2::EOC_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.628, 5.978)), module, WaterTable2::POS_MODE_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.628, 8.295)), module, WaterTable2::MOD_MODE_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(26.398, 63.259)), module, WaterTable2::ADDITIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 63.278)), module, WaterTable2::INTEGRAL_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 63.278)), module, WaterTable2::INTEGRAL_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.52, 63.278)), module, WaterTable2::ADDITIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 66.2)), module, WaterTable2::DIFFERENTIAL_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 66.2)), module, WaterTable2::DIFFERENTIAL_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(15.033, 69.123)), module, WaterTable2::SINC_INPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(44.155, 69.123)), module, WaterTable2::SINC_INPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.52, 69.123)), module, WaterTable2::MULTIPLICATIVE_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(26.398, 69.141)), module, WaterTable2::MULTIPLICATIVE_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(11.437, 123.322)), module, WaterTable2::INTEGRAL_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(17.915, 123.322)), module, WaterTable2::DIFFERENTIAL_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(24.393, 123.322)), module, WaterTable2::SINC_OUTPUT_L_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(40.383, 123.322)), module, WaterTable2::INTEGRAL_OUTPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(46.861, 123.322)), module, WaterTable2::DIFFERENTIAL_OUTPUT_R_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(53.339, 123.322)), module, WaterTable2::SINC_OUTPUT_R_LIGHT));

	}

	void appendContextMenu(Menu* menu) override {
		WaterTable2* module = dynamic_cast<WaterTable2*>(this->module);
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


Model* modelWaterTable2 = createModel<WaterTable2, WaterTable2Widget>("FreeSurface-WaterTable2");
