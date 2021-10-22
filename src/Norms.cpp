#include "plugin.hpp"


struct Norms : Module {
	enum ParamIds {
		LP_NORM_PARAM,
		LP_NORM_CV_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		LP_NORM_CV_INPUT,
		INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		NUM_LIGHTS
	};

	Norms() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(LP_NORM_PARAM, 0.05f, 32.f, 2.f, "Lp-norm exponent");
		configParam(LP_NORM_CV_PARAM, 0.f, 1.f, 0.f, "CV amount");
	}

	void process(const ProcessArgs& args) override {

		int channels = std::max(1, inputs[INPUT].getChannels());

		float lp_norm_cv = math::rescale(inputs[LP_NORM_CV_INPUT].getVoltage(0), -5.f, 5.f, -1.f, 1.f);
		float lp_norm_param_val = params[LP_NORM_PARAM].getValue();
		float lp_norm_param_cv_val = params[LP_NORM_CV_PARAM].getValue();

		float lp_norm = 0.0;
		float power = math::clamp(lp_norm_param_val + lp_norm_cv * lp_norm_param_cv_val, 0.05f, 8.f);
		for (int c = 0; c < channels; c++) {
			float val = math::rescale(inputs[INPUT].getVoltage(c), -5.f, 5.f, -1.f, 1.f);
			lp_norm += std::pow(std::abs(val), power);
		}

		lp_norm = std::pow(lp_norm, 1.f / power);

		if (lp_norm == 0.0f) {
			for (int c = 0; c < channels; c++) {
				outputs[OUTPUT].setVoltage(0.f, c);
			}
		} else {
			for (int c = 0; c < channels; c++) {
				float val = inputs[INPUT].getVoltage(c);
				outputs[OUTPUT].setVoltage(val / lp_norm, c);
			}
		}

		outputs[OUTPUT].setChannels(channels);
		
	}
};


struct NormsWidget : ModuleWidget {
	NormsWidget(Norms* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Norms.svg")));

		addParam(createParamCentered<MediumSmallBlobKnob>(mm2px(Vec(5.08, 20.591)), module, Norms::LP_NORM_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(5.08, 43.144)), module, Norms::LP_NORM_CV_PARAM));

		addInput(createInputCentered<PinkPort>(mm2px(Vec(5.08, 34.836)), module, Norms::LP_NORM_CV_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(5.08, 60.553)), module, Norms::INPUT));

		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(5.08, 75.098)), module, Norms::OUTPUT));
	}
};


//Model* modelNorms = createModel<Norms, NormsWidget>("FreeSurface-Norms");