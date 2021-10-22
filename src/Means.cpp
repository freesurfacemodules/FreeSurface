#include "plugin.hpp"

using simd::float_4;

struct RunningMean {
	bool first_sample = true;
	float mean = 0.0f;
	float population_variance = 0.0f;
	float time_constant = 0.1f;
	float inv_time_constant = 1.0f - time_constant;
	float stdev = 1.0f;

	void setTimeConstant(float tc, float cvScale, float cv) {
		time_constant = std::pow(2.0f, - 1.0f - 31.0f * simd::clamp(tc + cvScale * cv, 0.f, 1.f));
		inv_time_constant = 1.0f - time_constant;
	}

	void updateOne(float x) {
		if (first_sample) {
			mean = x;
			population_variance = 0.0;
			stdev = 1.0;
			first_sample = false;
			return;
		}
		float mean_old = mean;
		mean = inv_time_constant * mean + time_constant * x;
		population_variance = inv_time_constant * population_variance + time_constant * (x - mean_old) * (x - mean);
		stdev = std::sqrt(population_variance);
	}

	float mean_batch = 0.0f;
	float M2_batch = 0.0f;
	float count_batch = 0.0f;
	float X2_batch = 0.0f;
	float X_batch = 0.0f;

	void updateBatch(float x) {
		count_batch += 1.0f;
		float delta = x - mean_batch;
		mean_batch += delta / count_batch;
		float delta2 = x - mean_batch;
		M2_batch += delta * delta2;
		X2_batch += x * x;
		X_batch += x;
	}

	void finalizeBatch() {
		float new_population_variance = M2_batch / count_batch;
		//float new_mean = mean_batch;
		float new_mean = X_batch / count_batch;
		float X2 = X2_batch / count_batch;
		count_batch = 0.0f;
		mean_batch = 0.0f;
		M2_batch = 0.0f;
		X2_batch = 0.0f;
		X_batch = 0.0f;
		if (first_sample) {
			mean = new_mean;
			population_variance = new_population_variance;
			stdev = std::sqrt(population_variance);
			first_sample = false;
			return;
		}

		//population_variance = inv_time_constant * population_variance + time_constant * new_population_variance;
		mean = inv_time_constant * mean + time_constant * new_mean;
		population_variance = inv_time_constant * population_variance + time_constant * std::abs(X2 - mean*mean);

		stdev = std::sqrt(population_variance);
	}

	float normalize(float x) {
		return (x - mean) / stdev;
	}

	float getMean() {
		return mean;
	}

	float getStdev() {
		return stdev;
	}

};


struct Means : Module {
	enum ParamIds {
		TIME_PARAM,
		TIME_CV_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		TIME_CV_INPUT,
		INPUT,
		TRIGGER_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		NORMALIZED_OUTPUT,
		MEAN_OUTPUT,
		STDEV_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		NUM_LIGHTS
	};

	RunningMean runningMean = RunningMean();
	TriggerableFloat updateTrigger = TriggerableFloat();

	Means() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(TIME_PARAM, 0.0f, 1.f, 0.1f, "Running mean time constant");
		configParam(TIME_CV_PARAM, 0.f, 1.f, 0.f, "CV amount");
	}

	void process(const ProcessArgs& args) override {
		int channels = std::max(1, inputs[INPUT].getChannels());

		float time_cv = math::rescale(inputs[TIME_CV_INPUT].getVoltage(0), -5.f, 5.f, -1.f, 1.f);
		float time_param_val = params[TIME_PARAM].getValue();
		float time_param_cv_val = params[TIME_CV_PARAM].getValue();
		bool poly_in = channels > 1;
		runningMean.setTimeConstant(time_param_val, time_cv, time_param_cv_val);

		bool update_trigger_connected = inputs[TRIGGER_INPUT].isConnected();
		bool input_connected = inputs[INPUT].isConnected();
		if (update_trigger_connected) {
			updateTrigger.setInput(maxAll(inputs[TRIGGER_INPUT]));
		}

		bool needs_update = input_connected 
							&& ((update_trigger_connected && (updateTrigger.isTriggered()))
							|| (!update_trigger_connected));

		if (needs_update) {
			if (poly_in) {
				for (int c = 0; c < channels; c++) {
					float val = inputs[INPUT].getVoltage(c);
					runningMean.updateBatch(val);
				}
				runningMean.finalizeBatch();
			} else {
				runningMean.updateOne(inputs[INPUT].getVoltage(0));
			}
		}

		for (int c = 0; c < channels; c++) {
			float val = inputs[INPUT].getVoltage(c);
			float normalized = runningMean.normalize(val);
			outputs[NORMALIZED_OUTPUT].setVoltage(normalized, c);
		}
		
		if (outputs[MEAN_OUTPUT].isConnected()) {
			outputs[MEAN_OUTPUT].setVoltage(runningMean.getMean(), 0);
		}
		if (outputs[STDEV_OUTPUT].isConnected()) {
			outputs[STDEV_OUTPUT].setVoltage(runningMean.getStdev(), 0);
		}

		outputs[NORMALIZED_OUTPUT].setChannels(channels);
		outputs[MEAN_OUTPUT].setChannels(1);
		outputs[STDEV_OUTPUT].setChannels(1);
		
	}
};


struct MeansWidget : ModuleWidget {
	MeansWidget(Means* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Means.svg")));

		addParam(createParamCentered<MediumSmallBlobKnob>(mm2px(Vec(5.08, 20.591)), module, Means::TIME_PARAM));
		addParam(createParamCentered<SmallBlobKnob>(mm2px(Vec(5.08, 43.144)), module, Means::TIME_CV_PARAM));

		addInput(createInputCentered<PinkPort>(mm2px(Vec(5.08, 34.836)), module, Means::TIME_CV_INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(5.08, 57.907)), module, Means::INPUT));
		addInput(createInputCentered<RedPort>(mm2px(Vec(5.08, 69.019)), module, Means::TRIGGER_INPUT));

		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(5.08, 80.39)), module, Means::NORMALIZED_OUTPUT));
		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(5.08, 92.476)), module, Means::MEAN_OUTPUT));
		addOutput(createOutputCentered<RedCrossPort>(mm2px(Vec(5.08, 104.233)), module, Means::STDEV_OUTPUT));
	}
};


//Model* modelMeans = createModel<Means, MeansWidget>("FreeSurface-Means");