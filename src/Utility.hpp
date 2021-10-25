#pragma once

#include "rack.hpp"

using namespace rack;

template <typename T>
struct Triggerable {
	T resetState = T::mask();
	T triggered = 0.f;

	void setInput(T input) {
		input = simd::rescale(input, 0.1f, 2.f, 0.f, 1.f);
		T on = (input >= 1.f);
		T off = (input <= 0.f);
		triggered = ~resetState & on;
		resetState = simd::ifelse(off, 0.f, resetState);
		resetState = simd::ifelse(on, T::mask(), resetState);
	}

	T getTriggerState() {
		return triggered;
	}
};

struct TriggerableFloat {
	bool resetState = true;
	bool triggered = false;

	void setInput(float input) {
		input = simd::rescale(input, 0.1f, 2.f, 0.f, 1.f);
		bool on = (input >= 1.f);
		bool off = (input <= 0.f);
		triggered = !resetState && on;
		resetState = off ? false : resetState;
		resetState = on ? true : resetState;
	}

	bool isTriggered() {
		return triggered;
	}
};

inline float maxAll(Port& input) {
	int channels = std::max(1, input.getChannels());
	float max;
	for (int c = 0; c < channels; c++) {
		if (c == 0) {
			max = input.getVoltage(0);
		} else {
			max = std::max(max, input.getVoltage(c));
		}
	}
	return max;
}

struct StereoDCBiasRemover {
	float L, R = 0.0;
	float decay = 0.0001;
	void remove(float & in_L, float & in_R) {
		L  = simd::crossfade(L, in_L, decay);
		R  = simd::crossfade(R, in_R, decay);
		in_L = in_L - L;
		in_R = in_R - R;
	}
	void reset() {
		L = 0.;
		R = 0.;
	}
};

struct FixedTimeExpSlewLimiter {
	float val = 0.0;
	float decay;
	bool init = true;
	FixedTimeExpSlewLimiter() : decay(0.1f) {}
	FixedTimeExpSlewLimiter(float _decay) : decay(_decay) {}
	void limit(float &in) {
		if (init) {
			init = false;
			val = in;
		} else {
			val  = simd::crossfade(val, in, decay);
			in = val;
		}
	}
	void reset() {
		val = 0.;
		init = true;
	}
};