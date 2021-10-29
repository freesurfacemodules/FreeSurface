#pragma once

#include "rack.hpp"
#include <cstring>
#include <string>

using namespace rack;

using simd::float_4;
using simd::int32_4;

#define DUMMY_CV 999

template <size_t PARAM, size_t INPUT_CV, size_t PARAM_CV>
struct CVParamInput {
	Module* module;
	std::string json_label;
	float min;
	float max;
	float shift; // for pitch conversion where the tone generator needs to be tuned
	float sample_rate_scale; // also for pitch conversion
    float post_scale; // used to correct timestep range after the fact without fixing the tuning shift
	enum ParamType {
		Default,
		Modulo,
		Exponential,
		Pitch 
	} paramType;

	float paramCacheIn;
	float paramCacheOut;
	bool dirty = true;
	CVParamInput() {}

	void config(Module* module, float min, float max, float def, std::string json_label, std::string label = "", std::string unit = "", float displayBase = 0.f, float displayMultiplier = 1.f, float displayOffset = 0.f) {
		module->configParam(PARAM, min, max, def, label, unit, displayBase, displayMultiplier, displayOffset);
		if (PARAM_CV != DUMMY_CV) {
			module->configParam(PARAM_CV, -1.0, 1.0, 0.0, label + " CV");
		}
		this->json_label = json_label;
		this->module = module;
		this->min = min;
		this->max = max;
		this->paramType = ParamType::Default;
	}

	void configModulo(Module* module, float max, float def, std::string json_label, std::string label = "", std::string unit = "", float displayBase = 0.f, float displayMultiplier = 1.f, float displayOffset = 0.f) {
		module->configParam(PARAM, -HUGE_VALF, HUGE_VALF, def, label, unit, displayBase, displayMultiplier, displayOffset);
		if (PARAM_CV != DUMMY_CV) {
			module->configParam(PARAM_CV, -1.0, 1.0, 0.0, label + " CV");
		}
		this->json_label = json_label;
		this->module = module;
		this->min = 0.0;
		this->max = max;
		this->paramType = ParamType::Modulo;
	}

	void configExp(Module* module, float min, float max, float def, std::string json_label, std::string label = "", std::string unit = "", float displayBase = 0.f, float displayMultiplier = 1.f, float displayOffset = 0.f) {
		module->configParam(PARAM, min, max, def, label, unit, displayBase, displayMultiplier, displayOffset);
		if (PARAM_CV != DUMMY_CV) {
			module->configParam(PARAM_CV, -1.0, 1.0, 0.0, label + " CV");
		}
		this->json_label = json_label;
		this->module = module;
		this->min = min;
		this->max = max;
		this->paramType = ParamType::Exponential;
	}

	void configPitch(Module* module, float post_scale, float sample_rate_scale, float shift, float param_min, float param_max, float val_max, float def, std::string json_label, std::string label = "", std::string unit = "", float displayBase = 0.f, float displayMultiplier = 1.f, float displayOffset = 0.f) {
		module->configParam(PARAM, param_min, param_max, def, label, unit, displayBase, displayMultiplier, displayOffset);
		if (PARAM_CV != DUMMY_CV) {
			module->configParam(PARAM_CV, -1.0, 1.0, 0.0, label + " CV");
		}
		this->json_label = json_label;
		this->module = module;
		this->min = 0.0;
		this->max = val_max;
		this->shift = shift;
		this->sample_rate_scale = sample_rate_scale;
        this->post_scale = post_scale;
		this->paramType = ParamType::Pitch;
	}

	float getExpValue(float cv, float param, float input) {
		// Get gain
		float gain_param = simd::rescale(param, min, max, -1.0, 1.0);

		float gain = simd::clamp(gain_param + cv * input, -1.0, 1.0);
		
		float res;
		if (dirty || paramCacheIn != gain) {
			dirty = false;
			paramCacheIn = gain;
			res = simd::rescale(std::pow(2.0, gain * 8.f), 0.00390625, 256.0, min, max); // scale from (2^-8, 2^8) to (min, max)
			paramCacheOut = res;
		}

		return paramCacheOut;
	}

	float getPitchValue(float cv, float param, float input) {
		// Get 1voct
		input = simd::rescale(input, -1.0, 1.0, -5.0, 5.0); // hacky

		float one_voct = shift + param + cv * input;
		
		float res;
		if (dirty || paramCacheIn != one_voct) {
			dirty = false;
			paramCacheIn = one_voct;
			// scale by post_scale to get a little more range without recalibrating. bad idea to do this.
			res = std::min(max * post_scale, max * sample_rate_scale * std::pow(2.0f, one_voct) / 256.0f); // scale from 2^8 to max
			paramCacheOut = res;
		}

		return paramCacheOut;
	}

	float getValue() {
		float input = simd::rescale(module->inputs[INPUT_CV].getVoltage(0), -5.0, 5.0, -1.0, 1.0);
		float cv;
		if (PARAM_CV == DUMMY_CV) {
			cv = 1.0;
		} else {
			cv = module->params[PARAM_CV].getValue();
		}
		float param = module->params[PARAM].getValue();
		switch(paramType) {
			case Modulo:
				return math::eucMod(max / 2.0 * param + cv * max * input, max);
				break;
			case Exponential:
				return getExpValue(cv, param, input);
				break;
			case Pitch:
				return getPitchValue(cv, param, input);
				break;
			default:
				return simd::clamp(param + cv * (max - min) * input, min, max);
				break;
		}
	}

	float getParamKnob() {
		return module->params[PARAM].getValue();
	}

	void setParamKnob(float val) {
		module->params[PARAM].setValue(val);
	}

	void setSampleRateScale(float sample_rate_scale) {
		this->sample_rate_scale = sample_rate_scale;

		dirty = true;
	}

	void dataToJson(json_t* rootJ) {
		json_object_set_new(rootJ, json_label.c_str(), json_real(getParamKnob()));
	}

	void dataFromJson(json_t* rootJ) {
		json_t* j_val = json_object_get(rootJ, json_label.c_str());
		if (j_val)
			setParamKnob(json_real_value(j_val));
	}
};