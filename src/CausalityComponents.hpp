#pragma once

#include "rack.hpp"

using namespace rack;

extern Plugin* pluginInstance;

struct BlankPort : app::SvgPort {
	BlankPort() {
		setSvg(APP->window->loadSvg(asset::system("res/ComponentLibrary/TL1105_1.svg")));
	}
};

struct RedPort : app::SvgPort {
	RedPort() {
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RedPort.svg")));
	}
};

struct RedCrossPort : app::SvgPort {
	RedCrossPort() {
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RedCrossPort.svg")));
	}
};

struct PinkPort : app::SvgPort {
	PinkPort() {
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/PinkPort.svg")));
	}
};

struct BigBlobKnob : app::SvgKnob {
	BigBlobKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/BigBlobKnob.svg")));
		shadow->opacity = 0.f;
	}
};

struct MediumBigBlobKnob : app::SvgKnob {
	MediumBigBlobKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/MediumBigBlobKnob.svg")));
		shadow->opacity = 0.f;
	}
};

struct SmallBlobKnob : app::SvgKnob {
	SmallBlobKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/SmallBlobKnob.svg")));
		shadow->opacity = 0.f;
	}
};

struct MediumSmallBlobKnob : app::SvgKnob {
	MediumSmallBlobKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/MediumSmallBlobKnob.svg")));
		shadow->opacity = 0.f;
	}
};
