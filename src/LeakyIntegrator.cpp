#include "plugin.hpp"


using simd::float_4;

const float MIN_DT = 1e-6f;
const float MAX_DT = 1.0f;
const float MIN_GL = 1e-6f;
const float MAX_GL = 0.2f;

template <typename T>
struct LeakyIntegrateAndFire {
	// {'V_th': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0, 'E_L': -75.0, 'tref': 2.0, 'T': 400.0, 'dt': 0.1}
	/*
	T tau_m = 10.0f;
	T V_th = -55.0f;
	T V_reset = -75.0f;
	T g_L = 10.0f;
	T V_init = -75.0f;
	T E_L = -75.0f;
	T tref = 2.0f;
	T t = 400.0f;
	T dt = 0.1f;*/
	T tau_m = 10.0f;
	T V_th = 5.0f;
	T V_reset = -5.0f;
	T g_L = 1.0f;
	T V_init = -5.0f;
	T E_L = -5.0f;
	T tref = 2.0f;
	T dt = 0.1f;
	T v = V_init;
	T tr = 0.f;
	T spike = 0.f;

	void setDt(T DT, T DT_CV_scale, T DT_CV, T sample_rate_scale) {
		dt = simd::clamp(sample_rate_scale * (DT + DT_CV_scale * DT_CV), MIN_DT, MAX_DT);
	}

	void setGl(T GL, T GL_CV_scale, T GL_CV) {
		g_L = 20.0f * simd::clamp(GL + GL_CV_scale * GL_CV, MIN_GL, MAX_GL);
	}

	void update(T Iinj) {
		T refractory = tr > 0.f;
		T over_threshold = v >= V_th;
		tr = simd::ifelse(refractory, tr - 1.f, tr);
		v = simd::ifelse(refractory, V_reset, v);
		spike = simd::ifelse(~refractory & over_threshold, T::mask(), 0.f);
		v = simd::ifelse(~refractory & over_threshold, V_reset, v);
		tr = simd::ifelse(~refractory & over_threshold, tref / dt, tr);

		T dv = (-(v - E_L) + (Iinj / g_L)) * (dt / tau_m);
		v += dv;
	}

	T getScaledV() {
		return simd::rescale(v, V_reset, V_th, -5.f, 5.f);
	}

	T getSpike() {
		return spike;
	}

	T light() {
		return spike & 1.f;
	}

};


struct LeakyIntegrator : Module {
	enum ParamIds {
		TIME_CONSTANT_PARAM,
		TIME_CONSTANT_CV_PARAM,
		RUNNING_AVG_PARAM,
		RUNNING_AVG_CV_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		PRE_SYNAPTIC_INPUT,
		POST_SYNAPTIC_INPUT,
		TIME_CONSTANT_CV_INPUT,
		RUNNING_AVG_CV_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		TRIGGER_OUTPUT,
		RUNNING_AVG_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		TOP_LIGHT,
		BOTTOM_LIGHT,
		NUM_LIGHTS
	};

	LeakyIntegrateAndFire<float_4> leakyIntegrateAndFires[4];
	dsp::ClockDivider lightDivider;

	LeakyIntegrator() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(TIME_CONSTANT_PARAM, MIN_DT, MAX_DT, 0.1f, "Time Constant");
		configParam(RUNNING_AVG_PARAM, MIN_GL, MAX_GL, 0.1f, "Moving Average Time Constant");
		lightDivider.setDivision(16);
	}

	void process(const ProcessArgs& args) override {
		float timeConstant = params[TIME_CONSTANT_PARAM].getValue();
		float timeConstantCVScale = params[TIME_CONSTANT_CV_PARAM].getValue();
		float runningAvgTC = params[RUNNING_AVG_PARAM].getValue();
		float runningAvgTCCVScale = params[RUNNING_AVG_CV_PARAM].getValue();


		int channels = std::max(1, inputs[PRE_SYNAPTIC_INPUT].getChannels());

		int channels_time_constant_cv = std::max(1, inputs[TIME_CONSTANT_CV_INPUT].getChannels());
		int channels_running_avg_cv = std::max(1, inputs[RUNNING_AVG_CV_INPUT].getChannels());
		bool poly_time_constant_cv = false;
		bool poly_running_avg_cv = false;
		if (channels_time_constant_cv > 1 && channels_time_constant_cv == channels) {
			poly_time_constant_cv = true;
		}

		if (channels_running_avg_cv > 1 && channels_running_avg_cv == channels) {
			poly_running_avg_cv = true;
		}
		float sample_rate_scale = 44100.0f / args.sampleRate;
		for (int c = 0; c < channels; c += 4) {
			auto* leakyIntegrateAndFire = &leakyIntegrateAndFires[c / 4];

			float_4 time_constant_cv;
			if (poly_time_constant_cv) {
				time_constant_cv = simd::rescale(inputs[TIME_CONSTANT_CV_INPUT].getPolyVoltageSimd<float_4>(c), -5.f, 5.f, -1.f, 1.f);
			} else {
				time_constant_cv = simd::rescale(float_4(inputs[TIME_CONSTANT_CV_INPUT].getVoltage(0)), -5.f, 5.f, -1.f, 1.f);	
			}
			leakyIntegrateAndFire->setDt(float_4(timeConstant), float_4(timeConstantCVScale), time_constant_cv, sample_rate_scale);

			float_4 running_avg_cv;
			if (poly_running_avg_cv) {
				running_avg_cv = simd::rescale(inputs[RUNNING_AVG_CV_INPUT].getPolyVoltageSimd<float_4>(c), -5.f, 5.f, -1.f, 1.f);
			} else {
				running_avg_cv = simd::rescale(float_4(inputs[RUNNING_AVG_CV_INPUT].getVoltage(0)), -5.f, 5.f, -1.f, 1.f);
			}
			leakyIntegrateAndFire->setGl(float_4(runningAvgTC), float_4(runningAvgTCCVScale), running_avg_cv);
			

			float_4 Iinj = inputs[PRE_SYNAPTIC_INPUT].getPolyVoltageSimd<float_4>(c);


			leakyIntegrateAndFire->update(Iinj);


			if (outputs[TRIGGER_OUTPUT].isConnected()) {
				outputs[TRIGGER_OUTPUT].setVoltageSimd(10.f & leakyIntegrateAndFire->getSpike(), c);
			}

			if (outputs[RUNNING_AVG_OUTPUT].isConnected()) {
				outputs[RUNNING_AVG_OUTPUT].setVoltageSimd(leakyIntegrateAndFire->getScaledV(), c);
			}
		}

		outputs[TRIGGER_OUTPUT].setChannels(channels);
		outputs[RUNNING_AVG_OUTPUT].setChannels(channels);


		// Light
		if (lightDivider.process()) {
			if (channels == 1) {
				float lightValue = leakyIntegrateAndFires[0].light().s[0];
				lights[TOP_LIGHT].setSmoothBrightness(lightValue, args.sampleTime * lightDivider.getDivision());
				lights[BOTTOM_LIGHT].setSmoothBrightness(lightValue, args.sampleTime * lightDivider.getDivision());
			}
			else {
				float lightValue0 = leakyIntegrateAndFires[0].light().s[0];
				float lightValue1 = leakyIntegrateAndFires[1].light().s[0];
				lights[TOP_LIGHT].setSmoothBrightness(lightValue0, args.sampleTime * lightDivider.getDivision());
				lights[BOTTOM_LIGHT].setSmoothBrightness(lightValue1, args.sampleTime * lightDivider.getDivision());
			}
		}
	}
};

struct LeakyIntegratorWidget : ModuleWidget {
	LeakyIntegratorWidget(LeakyIntegrator* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/LeakyIntegrator.svg")));

		addParam(createParamCentered<BigBlobKnob>(mm2px(Vec(17.78, 22.224)), module, LeakyIntegrator::TIME_CONSTANT_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(26.346, 34.163)), module, LeakyIntegrator::TIME_CONSTANT_CV_PARAM));
		addParam(createParamCentered<BigBlobKnob>(mm2px(Vec(17.78, 84.335)), module, LeakyIntegrator::RUNNING_AVG_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(26.478, 96.745)), module, LeakyIntegrator::RUNNING_AVG_CV_PARAM));

		addInput(createInputCentered<PinkPort>(mm2px(Vec(9.293, 34.065)), module, LeakyIntegrator::TIME_CONSTANT_CV_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(17.78, 46.115)), module, LeakyIntegrator::PRE_SYNAPTIC_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(17.78, 62.391)), module, LeakyIntegrator::POST_SYNAPTIC_INPUT));
		addInput(createInputCentered<PinkPort>(mm2px(Vec(9.303, 96.666)), module, LeakyIntegrator::RUNNING_AVG_CV_INPUT));

		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(12.771, 115.682)), module, LeakyIntegrator::TRIGGER_OUTPUT));
		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(22.824, 115.683)), module, LeakyIntegrator::RUNNING_AVG_OUTPUT));

		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(8.612, 45.083)), module, LeakyIntegrator::TOP_LIGHT));
		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(26.541, 103.808)), module, LeakyIntegrator::BOTTOM_LIGHT));
	}
};


//Model* modelLeakyIntegrator = createModel<LeakyIntegrator, LeakyIntegratorWidget>("FreeSurface-LeakyIntegrator");
