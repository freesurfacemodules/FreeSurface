// KalmanPitchTracker.cpp
// Based on “Improved Real-Time Monophonic Pitch Tracking with the Extended Complex Kalman Filter”
// Das, Smith, and Chafe, JAES (2020)
//
// This module implements an extended complex Kalman filter pitch tracker.
// It takes a monophonic audio signal as input and outputs the estimated pitch in Hz.
// (A simplified silent‐frame and pitch-jump detection is used.)

#include "plugin.hpp"
#include <complex>
#include <cmath>

// Convenience alias for complex numbers
using Complex = std::complex<float>;
const float TWO_PI = 6.28318530718f;

struct KalmanPitchTracker : Module {
	enum ParamIds {
		SILENCE_THRESH_PARAM,     // threshold for detecting silence
		PITCH_JUMP_THRESH_PARAM,  // threshold (Hz) for rapid pitch change reinit
		PROCESS_NOISE_PARAM,      // base process noise coefficient
		SIGMA_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		AUDIO_INPUT,
		RESET_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		PITCH_OUTPUT,  // pitch output in Hz (CV scaled)
		AMP_OUTPUT,    // amplitude envelope (optional)
		FEST_OUTPUT,
		P_OUTPUT,
		K_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		NUM_LIGHTS
	};
	
	float f0 = 1.;
	float amp = 0.001;
	float phi = 0.001*M_PIf;
	
	const Complex I = Complex(0.f, 1.f);

	// State vector: x = [ x[0] = α, x[1] = u, x[2] = u* ]
	Complex x[3];
	Complex x_prev[3];
	Complex x_next[3];
	// Error covariance matrix P (3x3); stored row-major.
	Complex P[3][3];
	Complex P_prev[3][3];
	Complex P_next[3][3];

	// For reinitialization detection
	float prevF = 0.f;
	int silenceCount = 0;

	// Constants
	const float sigmaV = 1.f;  // measurement noise variance (fixed)
	// Sample time (set each step)
	float Ts = 1.f / APP->engine->getSampleRate();

	// Constructor
	KalmanPitchTracker() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(SILENCE_THRESH_PARAM, 0.f, 1.f, 0.01f, "Silence threshold");
		configParam(PITCH_JUMP_THRESH_PARAM, 0.f, 500.f, 50.f, "Max pitch jump (Hz)");
		configParam(PROCESS_NOISE_PARAM, 1.0f, 12.0f, 7.0f, "Process noise base");
		configParam(SIGMA_PARAM, 0.0f, 8.0f, 1.0, "Sigma");

		// Initialize state: assume zero pitch (α = exp(j0)=1) and no amplitude.
		x_prev[0] = Complex(1.f, 0.f); // corresponds to f = 0
		x_prev[1] = Complex(0.f, 0.f);
		x_prev[2] = Complex(0.f, 0.f);

		// Initialize covariance to large uncertainty
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
    			if (i==j) {
    			    P_prev[i][j] = Complex(1.f, 0.f);
    			} else {
    				P_prev[i][j] = Complex(0.f, 0.f);
    			}
			}
		}
	}

	// Helper: Reset state (e.g., on silence or pitch jump)
	void reinitialize(float f0, float amp = 0.f, float phase = 0.f) {
		float angle = TWO_PI * f0 * Ts;
		x_prev[0] = Complex(cosf(angle), sinf(angle));
		x_prev[1] = Complex(amp * cosf(angle + phase), amp * sinf(angle + phase));
		x_prev[2] = Complex(amp * cosf(angle + phase), -amp * sinf(angle + phase));

		// Initialize covariance to large uncertainty
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
    			if (i==j) {
    			    P_prev[i][j] = Complex(1.f, 0.f);
    			} else {
    				P_prev[i][j] = Complex(0.f, 0.f);
    			}
			}
		}
		prevF = f0;
		silenceCount = 0;
	}

	// Matrix multiplication helpers:
	void matMul33(const Complex A[3][3], const Complex B[3][3], Complex C[3][3]) {
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				C[i][j] = Complex(0.f, 0.f);
				for (int k = 0; k < 3; k++){
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}
	void matMul31(const Complex A[3][3], const Complex v[3], Complex r[3]) {
		for (int i = 0; i < 3; i++){
			r[i] = Complex(0.f, 0.f);
			for (int j = 0; j < 3; j++){
				r[i] += A[i][j] * v[j];
			}
		}
	}
	void matConjTranspose(const Complex in[3][3], Complex out[3][3]) {
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				out[i][j] = std::conj(in[j][i]);
			}
		}
	}

	// Process function (called each sample)
	void process(const ProcessArgs &args) override {
		Ts = 1.f / args.sampleRate;
		float in = inputs[AUDIO_INPUT].getVoltage();
		float sigmaP = params[SIGMA_PARAM].getValue();

		if (inputs[RESET_INPUT].isConnected() && (inputs[RESET_INPUT].getVoltage() > 1.f)) {
			reinitialize(1.f, 0.f, 0.f);
			outputs[PITCH_OUTPUT].setVoltage(0.f);
			outputs[AMP_OUTPUT].setVoltage(0.f);
			return;
		}

		float silenceThresh = params[SILENCE_THRESH_PARAM].getValue();
		if (fabsf(in) < silenceThresh) {
			silenceCount++;
			if (silenceCount > 64) {
				reinitialize(1.f, 0.f, 0.f);
				outputs[PITCH_OUTPUT].setVoltage(0.f);
				outputs[AMP_OUTPUT].setVoltage(0.f);
				return;
			}
		} else {
			silenceCount = 0;
		}



		float coeff = params[PROCESS_NOISE_PARAM].getValue();

		// K = (P_*H.adjoint()) * (H*P_*H.adjoint()+R).inverse();
		Complex S = 1.0f/(0.25f*(P[1][1] + 2.f*P[1][2] + P[2][2]) + sigmaP);
		Complex K_gain[3];
		for (int i = 0; i < 3; i++){
			Complex sum(0.f, 0.f);
			for (int j = 1; j < 3; j++){
				float H_j = 0.5f;
				sum += P_prev[i][j] * H_j;
			}
			K_gain[i] = sum * S;
		}
		
		// P = P_ - K*H*P_;
		Complex KH[3][3];
		// Compute (K*H) for each element: here H = [0., 0.5, 0.5]
		for (int i = 0; i < 3; i++){
			KH[i][0] = 0.0;
			KH[i][1] = K_gain[i] * 0.5f;
			KH[i][2] = K_gain[i] * 0.5f;
		}
		Complex I_KH[3][3];
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				I_KH[i][j] = (i == j) ? Complex(1.f, 0.f) - KH[i][j] : -KH[i][j];
			}
		}
		matMul33(I_KH, P_prev, P);

		//x = x_ + K*(input - H*x_);
		Complex r = in - 0.5f*(x_prev[1] + x_prev[2]);
		for (int i = 0; i < 3; i++){
			x[i] = x_prev[i] + K_gain[i] * r;
		}
		
		//x_next << x(0), x(0)*x(1), x(2)/x(0);
		x_next[0] = x[0];
		x_next[1] = x[0] * x[1];
		x_next[2] = x[2] / x[0];
		
		//F << 1,0,0, x(1),x(0),0, -x(2)/(x(0)*x(0)),0, unity/x(0);
		Complex F[3][3];
		F[0][0] = Complex(1.f, 0.f); F[0][1] = Complex(0.f, 0.f); F[0][2] = Complex(0.f, 0.f);
		F[1][0] = x[1];              F[1][1] = x[0];              F[1][2] = Complex(0.f, 0.f);
		F[2][0] = -x[2]/(x[0]*x[0]); F[2][1] = Complex(0.f, 0.f); F[2][2] = Complex(1.f, 0.f)/x[0];
		
		//D = input - H*x;
		Complex D = in - 0.5f*(x[1] + x[2]);
		
		//Q = pow(10,-(coeff-std::abs(D(0))));
		float Q = pow(10.f,-std::clamp(coeff-std::abs(D), 1.f, 20.f));
		//float Q = pow(10.f,-(coeff-std::abs(D)));
		
		//P_next = F*P*F.adjoint() + Q*Iden;
		Complex FP[3][3];
		matMul33(F, P, FP);
		Complex FconjT[3][3];
		matConjTranspose(F, FconjT);
		matMul33(FP, FconjT, P_next);
		P_next[0][0] += Q;
		P_next[1][1] += Q;
		P_next[2][2] += Q;	
			
		// f0 = std::abs(std::log(x(0))/(PI*2*I*Ts));
		Complex f0d = (TWO_PI * I * Ts);
		f0 = std::abs(std::log(x[0])/f0d);

		float pitchJumpThresh = params[PITCH_JUMP_THRESH_PARAM].getValue();
		if (fabsf(f0 - prevF) > pitchJumpThresh) {
			//reinitialize(261.f, std::abs(x[1]), std::arg(x[1]));
			reinitialize(f0, 0.5f, 0.f);
		}
		prevF = f0;

		// Output: scale pitch (Hz) to CV (1 V/octave with C4=261.63 Hz)
		float cv = (f0 > 0.f) ? log2f(f0 / 261.63f) : -10.f;
		outputs[PITCH_OUTPUT].setVoltage(cv);
		outputs[AMP_OUTPUT].setVoltage(std::abs(std::sqrt(x[1]*x[2])));
		outputs[FEST_OUTPUT].setVoltage(x[0].real());
		
		outputs[P_OUTPUT].setChannels(9);
		for (int i = 0; i < 3; i++){
			x_prev[i] = x_next[i];
			for (int j = 0; j < 3; j++){
				P_prev[i][j] = P_next[i][j];
				outputs[P_OUTPUT].setVoltage(std::abs(P_next[i][j]), 3*j+i);
			}
		}
		
		outputs[K_OUTPUT].setChannels(6);
		for (int i = 0; i < 3; i++) {
		    outputs[K_OUTPUT].setVoltage(K_gain[i].real(), 2*i);
			outputs[K_OUTPUT].setVoltage(K_gain[i].imag(), 2*i+1);			
		}
	}
};

struct KalmanPitchTrackerWidget : ModuleWidget {
	KalmanPitchTrackerWidget(KalmanPitchTracker* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Causality.svg")));
		addParam(createParam<RoundBlackKnob>(Vec(30, 50), module, KalmanPitchTracker::SILENCE_THRESH_PARAM));
		addParam(createParam<RoundBlackKnob>(Vec(30, 80), module, KalmanPitchTracker::PITCH_JUMP_THRESH_PARAM));
		addParam(createParam<RoundBlackKnob>(Vec(30, 110), module, KalmanPitchTracker::PROCESS_NOISE_PARAM));
		addParam(createParam<RoundBlackKnob>(Vec(30, 140), module, KalmanPitchTracker::SIGMA_PARAM));
		addInput(createInput<PJ301MPort>(Vec(30, 230), module, KalmanPitchTracker::AUDIO_INPUT));
		addInput(createInput<PJ301MPort>(Vec(30, 270), module, KalmanPitchTracker::RESET_INPUT));
		addOutput(createOutput<PJ301MPort>(Vec(30, 310), module, KalmanPitchTracker::PITCH_OUTPUT));
		addOutput(createOutput<PJ301MPort>(Vec(30, 350), module, KalmanPitchTracker::AMP_OUTPUT));
		addOutput(createOutput<PJ301MPort>(Vec(60, 310), module, KalmanPitchTracker::FEST_OUTPUT));
		
		addOutput(createOutput<PJ301MPort>(Vec(60, 50), module,  KalmanPitchTracker::P_OUTPUT));
		addOutput(createOutput<PJ301MPort>(Vec(60, 80), module,  KalmanPitchTracker::K_OUTPUT));
	}
};

Model* modelKalmanPitchTracker = createModel<KalmanPitchTracker, KalmanPitchTrackerWidget>("FreeSurface-KalmanPitchTracker");
