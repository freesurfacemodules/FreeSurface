#pragma once

#include "rack.hpp"
#include <cstring>

using namespace rack;

using simd::float_4;
using simd::int32_4;

// Just disables quadratic bezier interpolation 
// so we can see the raw output better
//#define DRAW_DEBUG

// These button definitions are super repetitive and need to be
// templated better, with just a function pointer or something
template <class TModule>
struct WaterTableModeButton : FreeSurfaceLogoToggleDark {
    TModule* module;
    WaterTableModeButton() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
		//ParamWidget::onButton(e);

        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->setNextModel();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableAdditiveModeLToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableAdditiveModeLToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleAdditiveModeL();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableAdditiveModeRToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableAdditiveModeRToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleAdditiveModeR();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableInputProbeTypeLToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableInputProbeTypeLToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleInputProbeTypeL();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableInputProbeTypeRToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableInputProbeTypeRToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleInputProbeTypeR();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableOutputProbeTypeLToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableOutputProbeTypeLToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleOutputProbeTypeL();
			e.consume(this);
		}
	}
};

template <class TModule>
struct WaterTableOutputProbeTypeRToggle : VektronixRoundToggleDark {
    TModule* module;
    WaterTableOutputProbeTypeRToggle() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->waveChannel.toggleOutputProbeTypeR();
			e.consume(this);
		}
	}
};

template <class TModule, size_t CHANNEL_SIZE, size_t CHANNEL_SIZE_FLOATS>
struct WaterTableDisplay : TransparentWidget {
	TModule* module;
	const float RADIUS = 0.8;
	const float MOD_RING_R = 0.5;
	const float Y_OFFSET = -6.0;
	Rect b;
	std::shared_ptr<Font> font;
	const NVGcolor orange_red_bright = nvgRGBA(0xf5, 0x39, 0x0a, 0xff);
	const NVGcolor orange_red = nvgRGBA(0xd0, 0x28, 0x0a, 0xff);
	const NVGcolor ember_orange = nvgRGBA(0xff, 0xcf, 0x3f, 0xff);
	const NVGcolor hot_white = nvgRGBA(0xff, 0xff, 0xeb, 0xff);
	const NVGcolor dark_grey = nvgRGBA(0x10, 0x10, 0x10, 0xff);
	const int HISTORY_SIZE = 16;
	std::deque<std::vector<float_4>> history;


	NVGcolor gradient(float x) {
		return nvgLerpRGBA(
			nvgLerpRGBA(orange_red, ember_orange, x),
			nvgLerpRGBA(ember_orange, hot_white, x),
			x);
	}

	WaterTableDisplay() : history(HISTORY_SIZE,
    		std::vector<float_4>(CHANNEL_SIZE, float_4::zero())) {
		font = APP->window->loadFont(asset::plugin(pluginInstance, "res/fixedsys-excelsior-301.ttf"));
	}

	void setBBox() {
		b = Rect(Vec(0, 0), box.size);
	}

	Vec scaleToBoxByX(Vec v) {
		Vec p;
		p.x = rescale(v.x, -1.f, 1.f, b.pos.x, b.pos.x + b.size.x);
		p.y = rescale(v.y, -1.f, 1.f, Y_OFFSET+b.pos.y, Y_OFFSET+b.pos.y + b.size.x); // circle y scaled to x size
		return p;
	}

	Vec circle(float i, float rad) {
		const float divs = 2.0 * M_PI / CHANNEL_SIZE_FLOATS;
		return Vec(rad * cos(divs * i - M_PI/2.),
				   rad * sin(divs * i - M_PI/2.));
	}

	Vec circle(float i) {
		return circle(i, RADIUS);
	}

	Vec getMarkerStartFromPos(float pos) {	
		Vec v = circle(pos);
		return scaleToBoxByX(v);
	}

	Vec getMarkerStartFromPos(float pos, float r) {	
		Vec v = circle(pos, r);
		return scaleToBoxByX(v);
	}

	Vec getScaledCircleNormal(float pos) {	
		Vec v = circle(pos);
		v = Vec(v.y,-v.x);
		return scaleToBoxByX(v);
	}

	float scaledBufferFromIndex(std::vector<float_4> &buffer, int i) {
		int f_in = i % 4;
		int f4_in = i / 4;
		return -rescale(buffer[f4_in][f_in],-10.0,10.0,-1.0,1.0);
	}


	void drawWaveform(const DrawArgs& args, std::vector<float_4> &buffer, int index) {
		float step = static_cast<float>(index) / static_cast<float>(HISTORY_SIZE-1);
		nvgScissor(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y);
		nvgBeginPath(args.vg);


		#ifdef DRAW_DEBUG
		for (unsigned int i = 0; i < CHANNEL_SIZE_FLOATS; i+=2) {
			Vec v0 = circle(i);
			float s0 =  scaledBufferFromIndex(buffer, i);
			//v0.y += s0;
			v0 = v0.plus(v0.mult(s0));
			Vec p0 = scaleToBoxByX(v0);
			if (i == 0) {
				nvgMoveTo(args.vg, p0.x, p0.y);
			} else {
				nvgLineTo(args.vg, p0.x, p0.y);
			}
		}
		#else
		for (unsigned int i = 0; i < CHANNEL_SIZE_FLOATS; i+=2) {
			Vec v0 = circle(i);
			Vec v1 = circle(i+1.0);
			float s0 =  scaledBufferFromIndex(buffer, i);
			float s1 =  scaledBufferFromIndex(buffer, i+1);
			v0 = v0.plus(v0.mult(s0));
			v1 = v1.plus(v1.mult(s1));

			Vec p0 = scaleToBoxByX(v0);
			Vec p1 = scaleToBoxByX(v1);

			if (i == 0) {
				nvgMoveTo(args.vg, p0.x, p0.y);
				nvgQuadTo(args.vg, p0.x, p0.y, p1.x, p1.y);
			} else {
				nvgQuadTo(args.vg, p0.x, p0.y, p1.x, p1.y);
			}
		}
		#endif
		nvgClosePath(args.vg);
		nvgPathWinding(args.vg, NVG_SOLID);

		for (unsigned int i = 0; i < CHANNEL_SIZE_FLOATS; i+=2) {
			Vec v0 = circle(i);
			Vec v1 = circle(i+1);
			Vec p0, p1;
			p0 = scaleToBoxByX(v0);
			p1 = scaleToBoxByX(v1);
			if (i == 0) {
				nvgMoveTo(args.vg, p0.x, p0.y);
				nvgQuadTo(args.vg, p0.x, p0.y, p1.x, p1.y);
			} else {
				nvgQuadTo(args.vg, p0.x, p0.y, p1.x, p1.y);
			}
		}

		nvgClosePath(args.vg);
		nvgPathWinding(args.vg, NVG_HOLE);

		const float alpha_scale = 1.0;
		float alpha = alpha_scale*simd::pow(step, 2.0);
		nvgFillColor(args.vg, nvgTransRGBAf(gradient(0.0), alpha));
		nvgFill(args.vg);

		nvgResetScissor(args.vg);
		
	}

	Vec pointToSegment(Vec v, Vec p, Vec q)
	{
		float pqx, pqy, dx, dy, d, t;
		pqx = q.x-p.x;
		pqy = q.y-p.y;
		dx = v.x-p.x;
		dy = v.y-p.y;
		d = pqx*pqx + pqy*pqy;
		t = pqx*dx + pqy*dy;
		if (d > 0) t /= d;
		if (t < 0) t = 0;
		else if (t > 1) t = 1;
		return Vec(p.x + t*pqx - v.x, dy = p.y + t*pqy - v.y);
	}

	Vec minBoxDistance(Vec pos, Vec top_l, Vec top_r, Vec bot_l, Vec bot_r) {
		Vec top = pointToSegment(pos, top_l, top_r);
		Vec left = pointToSegment(pos, top_l, bot_l);
		Vec right = pointToSegment(pos, top_r, bot_r);
		Vec bot = pointToSegment(pos, bot_l, bot_r);

		Vec min;
		min = (top.norm() < bot.norm()) ? top : bot;
		min = (min.norm() < left.norm()) ? min : left;
		min = (min.norm() < right.norm()) ? min : right;
		return min;
	}

	void drawTextBox(const DrawArgs& args, Vec pos_line, Vec center, const char* text) {

		const float width = 30.0;
		const float height = 10.0;
		Vec half = Vec(width/2., height/2.);
		Vec top_l = center.plus(Vec(-half.x, half.y));
		Vec top_r = center.plus(Vec( half.x, half.y));
		Vec bot_l = center.plus(Vec(-half.x,-half.y));
		Vec bot_r = center.plus(Vec( half.x,-half.y));

		Vec circToBox = minBoxDistance(pos_line, top_l, top_r, bot_l, bot_r);

		Vec text_pad = Vec(2,8);

		// There's no documentation anywhere on how compositing works
		// in NanoVG, and I'm pretty sure it doesn't work according to the specs
		// it references anyway, so just splatter this everywhere until it works properly
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgFillColor(args.vg, orange_red_bright);
		nvgStrokeColor(args.vg, orange_red_bright);
		
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgBeginPath(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgMoveTo(args.vg, top_l.x, top_l.y);
		nvgLineTo(args.vg, top_r.x, top_r.y);
		nvgLineTo(args.vg, bot_r.x, bot_r.y);
		nvgLineTo(args.vg, bot_l.x, bot_l.y);
		nvgLineTo(args.vg, top_l.x, top_l.y);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgClosePath(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgStroke(args.vg);
		nvgFill(args.vg);

		nvgStrokeColor(args.vg, orange_red_bright);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgBeginPath(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgMoveTo(args.vg, pos_line.x, pos_line.y);
		nvgLineTo(args.vg, pos_line.x + circToBox.x, pos_line.y + circToBox.y);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgStroke(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgFillColor(args.vg, dark_grey);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgText(args.vg, bot_l.x+text_pad.x, bot_l.y+text_pad.y, text, NULL);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgFill(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		

		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
	}

	void drawMarker(const DrawArgs& args, float pos, const char* text, bool left, bool up) {
		// Draw line
		nvgStrokeColor(args.vg, orange_red);
		nvgLineCap(args.vg, NVG_SQUARE);
		nvgStrokeWidth(args.vg, 1.0f);
		nvgFontSize(args.vg, 11);
		nvgFontFaceId(args.vg, font->handle);
		nvgFillColor(args.vg, orange_red_bright);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		{
			Vec p = getMarkerStartFromPos(pos);
			Vec n = getMarkerStartFromPos(pos, RADIUS*0.6);
			drawTextBox(args, p, n, text);
		}
		nvgStroke(args.vg);
	}

	float modRange(float pos, float width, float amp) {
		//return 2.0*(pos - (CHANNEL_SIZE_FLOATS / 2.0)) / CHANNEL_SIZE_FLOATS;

		amp = rack::simd::rescale(amp,-10.0,10.0,-1.0,1.0);
		float pos_to_mod_offset = 2.0*(pos - (CHANNEL_SIZE_FLOATS / 2.0)) / CHANNEL_SIZE_FLOATS;
		return width * (amp + pos_to_mod_offset) / 4.0;
	}

	void drawModInfo(const DrawArgs& args, float pos, float width, float amp) {
		float modInput = modRange(pos, width, amp);
		nvgBeginPath(args.vg);
		NVGcolor col = nvgLerpRGBA(orange_red, ember_orange, simd::pow(simd::abs(modInput), 2.0));
		nvgStrokeColor(args.vg, col);
		Vec center = Vec(0.0, 0.0);
		center = scaleToBoxByX(center);
		Vec side = Vec(MOD_RING_R, 0.0);
		side = scaleToBoxByX(side);
		float radius = side.minus(center).norm();
		float angle = M_PI * modInput;
		const float start_angle = M_PI/2.0;
		NVGwinding dir;
		if (math::sgn(angle) > 0) {
			dir = NVG_CW;
		} else {
			dir = NVG_CCW;
		}
		nvgArc(args.vg, center.x, center.y, radius, start_angle, start_angle + angle, dir);
		nvgStrokeWidth(args.vg, 5.0);
		nvgStroke(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);

	}

	void drawModelName(const DrawArgs& args) {
		Vec t_box = b.size.minus(Vec(0,20));
		Vec mid = Vec(b.size.x/2.0, 0.0);
		Vec p = t_box.plus(Vec(mid.x,8.0)).minus(Vec(b.size.x,0));


		nvgFillColor(args.vg, orange_red_bright);
		nvgBeginPath(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		nvgRect(args.vg, 0.0, t_box.y, b.size.x, 10.0);
		nvgClosePath(args.vg);
		nvgFill(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);

		nvgBeginPath(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgFillColor(args.vg, hot_white);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgFontSize(args.vg, 11);
		nvgFontFaceId(args.vg, font->handle);
		nvgTextAlign(args.vg, NVG_ALIGN_CENTER);
		const char* text = module->waveChannel.getModelString();
		nvgText(args.vg, p.x, p.y, text, NULL);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);
		nvgFill(args.vg);
		nvgGlobalCompositeOperation(args.vg, NVG_XOR);

	}

	void drawMarkers(const DrawArgs& args) {
		float pos_in_L = module->pos_in_L_param.getValue();
		float pos_in_R = module->pos_in_R_param.getValue();
		float pos_out_L = module->pos_out_L_param.getValue();
		float pos_out_R = module->pos_out_R_param.getValue();

		//nvgScissor(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y);

		const char* L_IN = "L_IN";
		const char* R_IN = "R_IN";
		const char* L_OUT = "L_OUT";
		const char* R_OUT = "R_OUT";
		drawMarker(args, pos_in_L, L_IN, true, true);
		if (module->waveChannel.isModMode()) {
			float amp_in_R = module->waveChannel.amp_in_R;
			float sig_in_R = module->sig_in_R_param.getValue();
			drawModInfo(args, pos_in_R, sig_in_R, amp_in_R);
		} else {
			drawMarker(args, pos_in_R, R_IN, false, true);
		}		
		drawMarker(args, pos_out_L, L_OUT, true, false);
		drawMarker(args, pos_out_R, R_OUT, false, false);

		//nvgResetScissor(args.vg);
	}

	void draw(const DrawArgs& args) override {
		if (!module)
			return;

		history.push_back(module->waveChannel.v_a0);
		history.pop_front();

		nvgBeginPath(args.vg);
		nvgRoundedRect(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y, 10.0);
		nvgFillColor(args.vg, dark_grey);
		nvgFill(args.vg);
		nvgClosePath(args.vg);

		nvgSave(args.vg);
			nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
			drawModelName(args);
		nvgRestore(args.vg);

		int i = 0;
		nvgSave(args.vg);
			nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
			for (auto v : history) {
				drawWaveform(args, v, i);
				i++;
			}
		nvgRestore(args.vg);

		nvgSave(args.vg);
			drawMarkers(args);
		nvgRestore(args.vg);

	}
};