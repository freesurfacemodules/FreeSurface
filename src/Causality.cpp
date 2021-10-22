#include "plugin.hpp"


using simd::float_4;


template <typename T>
struct CausalityDetector {
	T preTime = 0.f;
	T postTime = 0.f;
	T log2 = 0.69314718f;
	T invlog2 = 1.44269504f;
	T timeConstant = invlog2;
	T preResetState = T::mask();
	T postResetState = T::mask();
	T preTriggered = 0.f;
	T postTriggered = 0.f;
	T preAwait = 0.f;
	T postAwait = 0.f;
	T causality = 0.f;
	T updateTrigger = 0.f;
	T runningAvg = 0.f;
	T runningAvgTC = 0.95f;

	void setTimeConstant(T tc, T cvScale, T cv) {
		timeConstant = simd::clamp(tc + cvScale * cv, 1e-5f, 1.f) * invlog2;
	}

	void setRunningAvgTC(T tc, T cvScale, T cv) {
		runningAvgTC = simd::clamp(tc + cvScale * cv, 0.f, 1.f);
	}

	void setPresynapticInput(T input) {
		input = simd::rescale(input, 0.1f, 2.f, 0.f, 1.f);
		T on = (input >= 1.f);
		T off = (input <= 0.f);
		preTriggered = ~preResetState & on;
		preResetState = simd::ifelse(off, 0.f, preResetState);
		preResetState = simd::ifelse(on, T::mask(), preResetState);
	}

	void setPostsynapticInput(T input) {
		input = simd::rescale(input, 0.1f, 2.f, 0.f, 1.f);
		T on = (input >= 1.f);
		T off = (input <= 0.f);
		postTriggered = ~postResetState & on;
		postResetState = simd::ifelse(off, 0.f, postResetState);
		postResetState = simd::ifelse(on, T::mask(), postResetState);
	}

	void updatePreTime(float dt) {
		preTime = simd::ifelse(preAwait, preTime + dt, 0.f);
	}

	void updatePostTime(float dt) {
		postTime = simd::ifelse(postAwait, postTime + dt, 0.f);
	}

	void timeout() {
		T preTimeout = preTime > (timeConstant * 10.f);
		T postTimeout = postTime > (timeConstant * 10.f);
		preAwait = ~preTimeout & preAwait;
		postAwait = ~postTimeout & postAwait;
	}

	void updateCausality() {
		T preGotPost = preAwait & postTriggered;
		T postGotPre = postAwait & preTriggered;
		T coincident = preTriggered & postTriggered;	

		T preExp = simd::exp(- preTime / timeConstant) & (preGotPost & ~coincident);	
		T postExp = simd::exp(- postTime / timeConstant) & (postGotPre & ~coincident);

		T newCausality = preExp - postExp;
		T valChanged = (simd::abs(newCausality) > 0.0001f);
		updateTrigger = simd::ifelse(valChanged, 1.f, 0.f);
		causality = simd::ifelse(valChanged, newCausality, causality);
		runningAvg = simd::ifelse(valChanged, runningAvgTC * runningAvg + (1.0f - runningAvgTC) * newCausality, runningAvg);
	}

	void updateAwait() {
		preAwait = simd::ifelse(preTriggered, preTriggered, preAwait & ~postTriggered);
		postAwait = simd::ifelse(postTriggered, postTriggered, postAwait & ~preTriggered);
	}

	void step(float dt) {
		this->updatePreTime(dt);
		this->updatePostTime(dt);
		this->timeout();
		this->updateCausality();
		this->updateAwait();

	}

	T light0() {
		return preAwait & 1.f;
	}

	T light1() {
		return postAwait & 1.f;
	}
};


struct Causality : Module {
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

	CausalityDetector<float_4> causalityDetectors[4];
	dsp::ClockDivider lightDivider;

	Causality() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(TIME_CONSTANT_PARAM, 1e-5f, 1.f, .1f, "Time Constant");
		configParam(RUNNING_AVG_PARAM, 0.f, 1.f, .9f, "Moving Average Time Constant");
		lightDivider.setDivision(16);
	}

	void process(const ProcessArgs& args) override {
		float timeConstant = params[TIME_CONSTANT_PARAM].getValue();
		float timeConstantCVScale = params[TIME_CONSTANT_CV_PARAM].getValue();
		float runningAvgTC = params[RUNNING_AVG_PARAM].getValue();
		float runningAvgTCCVScale = params[RUNNING_AVG_CV_PARAM].getValue();


		int channels_pre = std::max(1, inputs[PRE_SYNAPTIC_INPUT].getChannels());
		int channels_post = std::max(1, inputs[POST_SYNAPTIC_INPUT].getChannels());
		int channels = std::max(channels_pre, channels_post);
		bool poly_pre = (channels_pre > 1);
		bool poly_post = (channels_post > 1);

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

		if (channels_pre > channels_post) {
			poly_post = false;
		}

		if (channels_post > channels_pre) {
			poly_pre = false;
		}

		for (int c = 0; c < channels; c += 4) {
			auto* causalityDetector = &causalityDetectors[c / 4];

			if (poly_time_constant_cv) {
				causalityDetector->setTimeConstant(float_4(timeConstant), float_4(timeConstantCVScale), inputs[TIME_CONSTANT_CV_INPUT].getPolyVoltageSimd<float_4>(c));
			} else {
				causalityDetector->setTimeConstant(float_4(timeConstant), float_4(timeConstantCVScale), float_4(inputs[TIME_CONSTANT_CV_INPUT].getVoltage(0)));	
			}

			if (poly_running_avg_cv) {
				causalityDetector->setRunningAvgTC(float_4(runningAvgTC), float_4(runningAvgTCCVScale), inputs[RUNNING_AVG_CV_INPUT].getPolyVoltageSimd<float_4>(c));
			} else {
				causalityDetector->setRunningAvgTC(float_4(runningAvgTC), float_4(runningAvgTCCVScale), float_4(inputs[RUNNING_AVG_CV_INPUT].getVoltage(0)));
			}

			

			float_4 presynapticInput, postsynapticInput;
			if (poly_post && !poly_pre) {
				presynapticInput = float_4(inputs[PRE_SYNAPTIC_INPUT].getVoltage(0));
				postsynapticInput = inputs[POST_SYNAPTIC_INPUT].getPolyVoltageSimd<float_4>(c);
			} else if (poly_pre && !poly_post) {
				presynapticInput = inputs[PRE_SYNAPTIC_INPUT].getPolyVoltageSimd<float_4>(c);
				postsynapticInput = float_4(inputs[POST_SYNAPTIC_INPUT].getVoltage(0));
			} else if (poly_pre && poly_post) {
				presynapticInput = inputs[PRE_SYNAPTIC_INPUT].getPolyVoltageSimd<float_4>(c);
				postsynapticInput = inputs[POST_SYNAPTIC_INPUT].getPolyVoltageSimd<float_4>(c);
			} else {
				presynapticInput = float_4(inputs[PRE_SYNAPTIC_INPUT].getVoltage(0));
				postsynapticInput = float_4(inputs[POST_SYNAPTIC_INPUT].getVoltage(0));
			}

			causalityDetector->setPresynapticInput(presynapticInput);
			causalityDetector->setPostsynapticInput(postsynapticInput);

			causalityDetector->step(args.sampleTime);

			// Outputs
			//if (outputs[CAUSALITY_OUTPUT].isConnected()) {
			//	outputs[CAUSALITY_OUTPUT].setVoltageSimd(5.f * causalityDetector->causality, c);
			//}

			if (outputs[TRIGGER_OUTPUT].isConnected()) {
				outputs[TRIGGER_OUTPUT].setVoltageSimd(10.f * causalityDetector->updateTrigger, c);
			}

			if (outputs[RUNNING_AVG_OUTPUT].isConnected()) {
				outputs[RUNNING_AVG_OUTPUT].setVoltageSimd(5.f * causalityDetector->runningAvg, c);
			}
		}

		//outputs[CAUSALITY_OUTPUT].setChannels(channels);
		outputs[TRIGGER_OUTPUT].setChannels(channels);
		outputs[RUNNING_AVG_OUTPUT].setChannels(channels);


		// Light
		if (lightDivider.process()) {
			if (channels == 1) {
				float lightValue0 = causalityDetectors[0].light0().s[0];
				float lightValue1 = causalityDetectors[0].light1().s[0];
				lights[TOP_LIGHT].setSmoothBrightness(lightValue0, args.sampleTime * lightDivider.getDivision());
				lights[BOTTOM_LIGHT].setSmoothBrightness(lightValue1, args.sampleTime * lightDivider.getDivision());
			}
			else {
				float lightValue0 = causalityDetectors[0].light0().s[0];
				float lightValue1 = causalityDetectors[0].light1().s[0];
				lights[TOP_LIGHT].setSmoothBrightness(lightValue0, args.sampleTime * lightDivider.getDivision());
				lights[BOTTOM_LIGHT].setSmoothBrightness(lightValue1, args.sampleTime * lightDivider.getDivision());
			}
		}
	}
};

struct CausalityWidget : ModuleWidget {
	CausalityWidget(Causality* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Causality.svg")));

		/*
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(15.427, 18.662)), module, Causality::TIME_CONSTANT_PARAM));
		addParam(createParamCentered<Trimpot>(mm2px(Vec(15.528, 66.884)), module, Causality::RUNNING_AVG_PARAM));

		addInput(createInputCentered<BlankPort>(mm2px(Vec(15.24, 39.379)), module, Causality::PRE_SYNAPTIC_INPUT));
		addInput(createInputCentered<BlankPort>(mm2px(Vec(15.24, 56.356)), module, Causality::POST_SYNAPTIC_INPUT));

		addOutput(createOutputCentered<BlankPort>(mm2px(Vec(15.482, 77.735)), module, Causality::RUNNING_AVG_OUTPUT));
		addOutput(createOutputCentered<BlankPort>(mm2px(Vec(15.482, 94.293)), module, Causality::CAUSALITY_OUTPUT));
		addOutput(createOutputCentered<BlankPort>(mm2px(Vec(15.482, 111.076)), module, Causality::TRIGGER_OUTPUT));

		addChild(createLight<SmallLight<RedGreenBlueLight>>(Vec(15.482, 2.0f), module, Causality::PHASE_LIGHT));*/

		addParam(createParamCentered<BigBlobKnob>(mm2px(Vec(17.78, 22.224)), module, Causality::TIME_CONSTANT_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(26.346, 34.163)), module, Causality::TIME_CONSTANT_CV_PARAM));
		addParam(createParamCentered<BigBlobKnob>(mm2px(Vec(17.78, 84.335)), module, Causality::RUNNING_AVG_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(26.478, 96.745)), module, Causality::RUNNING_AVG_CV_PARAM));

		addInput(createInputCentered<PinkPort>(mm2px(Vec(9.293, 34.065)), module, Causality::TIME_CONSTANT_CV_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(17.78, 46.115)), module, Causality::PRE_SYNAPTIC_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(17.78, 62.391)), module, Causality::POST_SYNAPTIC_INPUT));
		addInput(createInputCentered<PinkPort>(mm2px(Vec(9.303, 96.666)), module, Causality::RUNNING_AVG_CV_INPUT));

		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(12.771, 115.682)), module, Causality::TRIGGER_OUTPUT));
		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(22.824, 115.683)), module, Causality::RUNNING_AVG_OUTPUT));

		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(8.612, 45.083)), module, Causality::TOP_LIGHT));
		addChild(createLightCentered<SmallLight<RedLight>>(mm2px(Vec(26.541, 103.808)), module, Causality::BOTTOM_LIGHT));
	}
};


//Model* modelCausality = createModel<Causality, CausalityWidget>("FreeSurface-Causality");
