#pragma once

#include "rack.hpp"

using namespace rack;

extern Plugin* pluginInstance;

template <typename TBase>
struct VektronixDiskLight : RectangleLight<TBase> {
	VektronixDiskLight() {
		this->box.size = window::mm2px(math::Vec(3.0, 1.0));
	}
};

struct VektronixToggle : SvgSwitch {
	VektronixToggle() {
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixButtonUp.svg")));
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixButtonDown.svg")));
		shadow->opacity = 0.f;
	}
};

struct VektronixToggleDark : SvgSwitch {
	VektronixToggleDark() {
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixButtonUpDark.svg")));
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixButtonDownDark.svg")));
		shadow->opacity = 0.f;
	}
};

struct VektronixRoundToggleDark : SvgSwitch {
	VektronixRoundToggleDark() {
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixRoundButtonUpDark.svg")));
		addFrame(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixRoundButtonDownDark.svg")));
		shadow->opacity = 0.f;
	}
};

struct VektronixPort : app::SvgPort {
	VektronixPort() {
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixPort.svg")));
		shadow->opacity = 0.f;
	}
};

struct VektronixPortBorderlessDark : app::SvgPort {
	VektronixPortBorderlessDark() {
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixPortBorderlessDark.svg")));
		shadow->opacity = 0.f;
	}
};

struct VektronixBigKnob : app::SvgKnob {
	VektronixBigKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnob.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobCap.svg")));
		capFb->addChild(cap);
		addChild(capFb);
	}
};

struct RotatingIndicator : widget::TransparentWidget {
	widget::FramebufferWidget* fb;
	widget::TransformWidget* tw;
	widget::SvgWidget* sw;
	/** Angles in radians */
	float minAngle = 0.f;
	float maxAngle = M_PI;
	RotatingIndicator() {
		fb = new widget::FramebufferWidget;
		addChild(fb);

		tw = new widget::TransformWidget;
		fb->addChild(tw);

		sw = new widget::SvgWidget;
		tw->addChild(sw);
	}

	void setSvg(std::shared_ptr<Svg> svg) {
		sw->setSvg(svg);
		tw->box.size = sw->box.size;
		fb->box.size = sw->box.size;
		box.size = sw->box.size;
	}

	void rotateFromParent(float angle) {
		// Re-transform the widget::TransformWidget
		tw->identity();
		// Rotate SVG
		math::Vec center = sw->box.getCenter();
		tw->translate(center);
		tw->rotate(angle);
		tw->translate(center.neg());
		fb->dirty = true;
	}
};

struct VektronixIndicatorDark : RotatingIndicator {
	VektronixIndicatorDark() {
		minAngle = -M_PI;
		maxAngle = M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobCapIndicatorDark.svg")));
	}
};

struct VektronixIndicatorSmallDark : RotatingIndicator {
	VektronixIndicatorSmallDark() {
		minAngle = -M_PI;
		maxAngle = M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixSmallKnobCapIndicatorDark.svg")));
	}
};

struct VektronixIndicatorTinyDark : RotatingIndicator {
	VektronixIndicatorTinyDark() {
		minAngle = -M_PI;
		maxAngle = M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixTinyKnobCapIndicatorDark.svg")));
	}
};

static float angleFromParamQuantity(ParamQuantity* paramQuantity, float scale, float minAngle, float maxAngle) {
	float value = paramQuantity->getSmoothValue();
	float angle;
	if (!paramQuantity->isBounded()) {
		// just map from -1..1, different from default behavior
		angle = math::rescale(scale * paramQuantity->getValue(), -1.f, 1.f, minAngle, maxAngle);
	}
	else if (paramQuantity->getRange() == 0.f) {
		// Center angle for zero range
		angle = (minAngle + maxAngle) / 2.f;
	} else {
		// Proportional angle for finite range
		angle = math::rescale(value, paramQuantity->getMinValue(), paramQuantity->getMaxValue(), minAngle, maxAngle);
	}
	return std::fmod(angle, 2 * M_PI);
}

struct VektronixInfiniteBigKnob : app::SvgKnob {
	float scale = 1.0;
	VektronixIndicatorDark* indicator;
	VektronixInfiniteBigKnob() {
		minAngle = -M_PI;
		maxAngle = M_PI;
		speed = 1.0;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobDark.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobCapDark.svg")));
		capFb->addChild(cap);
		addChild(capFb);
		indicator = new VektronixIndicatorDark;
		addChild(indicator);
	}

	void onChange(const event::Change& e) override {
		ParamQuantity* paramQuantity = getParamQuantity();
		// Re-transform the widget::TransformWidget
		if (paramQuantity) {
			float angle = angleFromParamQuantity(paramQuantity, scale, minAngle, maxAngle);
			indicator->rotateFromParent(angle); // draw our little rotating indicator on top of the cap
			tw->identity();
			// Rotate SVG
			math::Vec center = sw->box.getCenter();
			tw->translate(center);
			tw->rotate(angle);
			tw->translate(center.neg());
			fb->dirty = true;
		}
		Knob::onChange(e);
	}

	// unbounded knobs don't reset or randomize,
	// so we need to override
	void onDoubleClick(const event::DoubleClick& e) override {
		ParamQuantity* paramQuantity = getParamQuantity();
		if (paramQuantity) {
			float oldValue = paramQuantity->getValue();

			float value = paramQuantity->getDefaultValue();
			paramQuantity->setValue(value);

			float newValue = paramQuantity->getValue();

			if (oldValue != newValue) {
				// Push ParamChange history action
				history::ParamChange* h = new history::ParamChange;
				h->name = "reset parameter";
				h->moduleId = paramQuantity->module->id;
				h->paramId = paramQuantity->paramId;
				h->oldValue = oldValue;
				h->newValue = newValue;
				APP->history->push(h);
			}
		}
	}

};

struct VektronixBigKnobDark : app::SvgKnob {
	float scale = 1.0;
	VektronixIndicatorDark* indicator;
	VektronixBigKnobDark() {
		minAngle = -0.83*M_PI;
		maxAngle = 0.83*M_PI;
		speed = 1.0;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobDark.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixBigKnobCapDark.svg")));
		capFb->addChild(cap);
		addChild(capFb);
		indicator = new VektronixIndicatorDark;
		addChild(indicator);
	}

	void onChange(const event::Change& e) override {

		ParamQuantity* paramQuantity = getParamQuantity();
		if (paramQuantity) {
			float angle = angleFromParamQuantity(paramQuantity, scale, minAngle, maxAngle);
			indicator->rotateFromParent(angle); // draw our little rotating indicator on top of the cap
			tw->identity();
			// Rotate SVG
			math::Vec center = sw->box.getCenter();
			tw->translate(center);
			tw->rotate(angle);
			tw->translate(center.neg());
			fb->dirty = true;
		}
		Knob::onChange(e);
	}

};

struct VektronixSmallKnobDark : app::SvgKnob {
	float scale = 1.0;
	VektronixIndicatorSmallDark* indicator;
	VektronixSmallKnobDark() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		speed = 1.0;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixSmallKnobDark.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixSmallKnobCapDark.svg")));
		capFb->addChild(cap);
		addChild(capFb);
		indicator = new VektronixIndicatorSmallDark;
		addChild(indicator);
	}

	void onChange(const event::Change& e) override {
		ParamQuantity* paramQuantity = getParamQuantity();
		// Re-transform the widget::TransformWidget
		if (paramQuantity) {
			float angle = angleFromParamQuantity(paramQuantity, scale, minAngle, maxAngle);
			indicator->rotateFromParent(angle); // draw our little rotating indicator on top of the cap
			tw->identity();
			// Rotate SVG
			math::Vec center = sw->box.getCenter();
			tw->translate(center);
			tw->rotate(angle);
			tw->translate(center.neg());
			fb->dirty = true;
		}
		Knob::onChange(e);
	}
};

struct VektronixTinyKnobDark : app::SvgKnob {
	float scale = 1.0;
	VektronixIndicatorTinyDark* indicator;
	VektronixTinyKnobDark() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		speed = 1.0;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixTinyKnobDark.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixTinyKnobCapDark.svg")));
		capFb->addChild(cap);
		addChild(capFb);
		indicator = new VektronixIndicatorTinyDark;
		addChild(indicator);
	}

	void onChange(const event::Change& e) override {
		ParamQuantity* paramQuantity = getParamQuantity();
		// Re-transform the widget::TransformWidget
		if (paramQuantity) {
			float angle = angleFromParamQuantity(paramQuantity, scale, minAngle, maxAngle);
			indicator->rotateFromParent(angle); // draw our little rotating indicator on top of the cap
			tw->identity();
			// Rotate SVG
			math::Vec center = sw->box.getCenter();
			tw->translate(center);
			tw->rotate(angle);
			tw->translate(center.neg());
			fb->dirty = true;
		}
		Knob::onChange(e);
	}
};

struct VektronixSmallKnob : app::SvgKnob {
	VektronixSmallKnob() {
		minAngle = -0.83 * M_PI;
		maxAngle = 0.83 * M_PI;
		setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixSmallKnob.svg")));
		// Add cap
		widget::FramebufferWidget* capFb = new widget::FramebufferWidget;
		widget::SvgWidget* cap = new widget::SvgWidget;
		cap->setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/VektronixSmallKnobCap.svg")));
		capFb->addChild(cap);
		addChild(capFb);
	}
};