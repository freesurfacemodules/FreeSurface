#pragma once

#include "rack.hpp"
#include <cstring>
#include <functional>
#include <iostream>

using namespace rack;

using simd::float_4;
using simd::int32_4;

// #define DEBUG_PROBE_WEIGHT

template <class TModule, size_t CHANNEL_SIZE, size_t CHANNEL_SIZE_FLOATS>
struct WaterTable2Display : TransparentWidget {
    // TODO these should be function or template parameters
    const unsigned int CHANNEL_SIZE_X = 8;
    const unsigned int CHANNEL_SIZE_Y = 8;

    const float x_off = 1./(static_cast<float>(CHANNEL_SIZE_X));
    const float y_off = 1./(static_cast<float>(CHANNEL_SIZE_Y));
    const float gridscale = 0.8;

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

	float halo_brightness = 0.5;

	const int HISTORY_SIZE = 16;
	std::deque<std::vector<float_4>> history;
	std::string fontPath;



	NVGcolor gradient(float x) {
		return nvgLerpRGBA(
			nvgLerpRGBA(orange_red, ember_orange, x),
			nvgLerpRGBA(ember_orange, hot_white, x),
			x);
	}

	WaterTable2Display() :
			history(HISTORY_SIZE, std::vector<float_4>(CHANNEL_SIZE, float_4::zero())),
			fontPath(asset::plugin(pluginInstance, "res/fixedsys-excelsior-301.ttf")) {}

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

	Vec getMarkerStartFromPos(Vec pos) {
		//Vec v = circle(pos);
		return scaleToBoxByX(pos);
	}

	Vec getMarkerStartFromPos(Vec pos, float r) {
		//Vec v = circle(pos, r);
        Vec v = pos.plus(Vec(1,1).mult(r));
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

    float scaledBufferFromXY(std::vector<float_4> &buffer, unsigned int i, unsigned int j, unsigned int channel_size_x, unsigned int channel_size_y) {
        int channel_size_f4 = channel_size_x / 4;
        int f_in = i % 4;
        int f4_in = i / 4;
        return -rescale(buffer[f4_in + j * channel_size_f4][f_in],-10.0,10.0,-1.0,1.0);
    }

	void drawWaveform(const DrawArgs& args, std::vector<float_4> &buffer, int index, const NVGpaint& p, bool halo) {
		float step = static_cast<float>(index) / static_cast<float>(HISTORY_SIZE-1);
        const float alpha_scale = 1.0;
        float alpha = alpha_scale*simd::pow(step, 2.0);
		nvgScissor(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y);

        #ifdef DEBUG_PROBE_WEIGHT
            const float displacement = 10.0f;
        #else
            const float displacement = 20.0f;
        #endif

        // -1 because we plot two points per iteration
        for (unsigned int j = 0; j < CHANNEL_SIZE_Y; j++) {
            nvgBeginPath(args.vg);
            for (unsigned int i = 0; i < (CHANNEL_SIZE_X - 1); i++) {
                const float xr0 = (static_cast<float>(i  ) / static_cast<float>(CHANNEL_SIZE_X)) - 0.5;
                const float xr1 = (static_cast<float>(i+1) / static_cast<float>(CHANNEL_SIZE_X)) - 0.5;
                const float xr2 = (static_cast<float>(i+2) / static_cast<float>(CHANNEL_SIZE_X)) - 0.5;
                const float yr0 = (static_cast<float>(j  ) / static_cast<float>(CHANNEL_SIZE_Y)) - 0.5;

                float xp  = rescale(xr0, -0.5, 0.5, -gridscale, gridscale) + x_off;
                float xp1 = rescale(xr1, -0.5, 0.5, -gridscale, gridscale) + x_off;
                float xp2 = rescale(xr2, -0.5, 0.5, -gridscale, gridscale) + x_off;
                float yp  = rescale(yr0, -0.5, 0.5, -gridscale, gridscale) + y_off;

                const Vec disp_vec0 = Vec(xr0,yr0).mult(displacement);
                const Vec disp_vec1 = Vec(xr1,yr0).mult(displacement);
                const Vec disp_vec2 = Vec(xr2,yr0).mult(displacement);

                Vec v0 = Vec(xp, yp);
                Vec v1 = Vec(xp1, yp);
                Vec v2 = Vec(xp2, yp);

                float s0 = scaledBufferFromXY(buffer, i, j, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);
                float s1 = scaledBufferFromXY(buffer, i + 1, j, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);
                float s2 = scaledBufferFromXY(buffer, (i + 2)%8, j, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);

                v0 = v0.plus(disp_vec0.mult(s0));
                v1 = v1.plus(disp_vec1.mult(s1));
                v2 = v2.plus(disp_vec2.mult(s2));

                Vec p0 = scaleToBoxByX(v0);
                Vec p1 = scaleToBoxByX(v1);
                Vec p2 = scaleToBoxByX(v2);

                // control points based on derivatives
                Vec cp0 = p0.plus(p1).mult(0.5);
                Vec cp1 = p1.mult(3.).minus(p2).mult(0.5);

                if (i == 0) {
                    nvgMoveTo(args.vg, p0.x, p0.y);
                    nvgBezierTo(args.vg, cp0.x, cp0.y, cp1.x, cp1.y, p1.x, p1.y);
                } else {
                    nvgBezierTo(args.vg, cp0.x, cp0.y, cp1.x, cp1.y, p1.x, p1.y);
                }
            }
            nvgStrokeColor(args.vg, nvgTransRGBAf(orange_red, alpha));
            nvgStrokeWidth(args.vg, 1.0);
            nvgStroke(args.vg);
        }

        // same as the first but moving vertically
        for (unsigned int i = 0; i < CHANNEL_SIZE_X; i++) {
            nvgBeginPath(args.vg);
            for (unsigned int j = 0; j < (CHANNEL_SIZE_Y - 1); j++) {
                const float yr0 = (static_cast<float>(j  ) / static_cast<float>(CHANNEL_SIZE_Y)) - 0.5;
                const float yr1 = (static_cast<float>(j+1) / static_cast<float>(CHANNEL_SIZE_Y)) - 0.5;
                const float yr2 = (static_cast<float>(j+2) / static_cast<float>(CHANNEL_SIZE_Y)) - 0.5;
                const float xr0 = (static_cast<float>(i  ) / static_cast<float>(CHANNEL_SIZE_X)) - 0.5;

                float yp  = rescale(yr0, -0.5, 0.5, -gridscale, gridscale) + y_off;
                float yp1 = rescale(yr1, -0.5, 0.5, -gridscale, gridscale) + y_off;
                float yp2 = rescale(yr2, -0.5, 0.5, -gridscale, gridscale) + y_off;
                float xp  = rescale(xr0, -0.5, 0.5, -gridscale, gridscale) + x_off;

                const Vec disp_vec0 = Vec(xr0,yr0).mult(displacement);
                const Vec disp_vec1 = Vec(xr0,yr1).mult(displacement);
                const Vec disp_vec2 = Vec(xr0,yr2).mult(displacement);

                Vec v0 = Vec(xp, yp);
                Vec v1 = Vec(xp, yp1);
                Vec v2 = Vec(xp, yp2);

                float s0 = scaledBufferFromXY(buffer, i, j, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);
                float s1 = scaledBufferFromXY(buffer, i, j+1, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);
                float s2 = scaledBufferFromXY(buffer, i, (j+2)%8, CHANNEL_SIZE_X, CHANNEL_SIZE_Y);

                v0 = v0.plus(disp_vec0.mult(s0));
                v1 = v1.plus(disp_vec1.mult(s1));
                v2 = v2.plus(disp_vec2.mult(s2));

                Vec p0 = scaleToBoxByX(v0);
                Vec p1 = scaleToBoxByX(v1);
                Vec p2 = scaleToBoxByX(v2);

                Vec cp0 = p0.plus(p1).mult(0.5);
                Vec cp1 = p1.mult(3.).minus(p2).mult(0.5);

                if (j == 0) {
                    nvgMoveTo(args.vg, p0.x, p0.y);
                    nvgBezierTo(args.vg, cp0.x, cp0.y, cp1.x, cp1.y, p1.x, p1.y);
                } else {
                    nvgBezierTo(args.vg, cp0.x, cp0.y, cp1.x, cp1.y, p1.x, p1.y);
                }
            }
            nvgStrokeColor(args.vg, nvgTransRGBAf(orange_red, alpha));
            nvgStrokeWidth(args.vg, 1.0);
            nvgStroke(args.vg);
        }

        // will figure out later what to do with the halo
		/*if (halo) {
			nvgFillPaint(args.vg, p);
			nvgAlpha(args.vg, 0.8);
		}*/

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

	void drawTextBoxHalo(const DrawArgs& args, Vec center) {
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		//const float width = 30.0;
		//const float height = 10.0;
		const float width = 45.0;
		const float height = 25.0;
		Vec half = Vec(width/2., height/2.);

		Vec bot_l = center.plus(Vec(-half.x,-half.y));

		nvgBeginPath(args.vg);
		NVGpaint p = nvgBoxGradient(args.vg, bot_l.x, bot_l.y, width, height, 12.5, 35.0,
				nvgTransRGBAf(orange_red_bright, halo_brightness),
				nvgTransRGBAf(orange_red_bright,-1.0f));
		nvgRoundedRect(args.vg, bot_l.x, bot_l.y, width, height, 10.0);
		nvgClosePath(args.vg);
		nvgFillPaint(args.vg, p);
		nvgFill(args.vg);
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

	void drawMarker(const DrawArgs& args, Vec pos, const char* text, bool left, bool up, std::shared_ptr<Font> font) {
		Vec p = getMarkerStartFromPos(pos);
		Vec n = getMarkerStartFromPos(pos, 0.2);

		nvgStrokeColor(args.vg, orange_red);
		nvgLineCap(args.vg, NVG_SQUARE);
		nvgStrokeWidth(args.vg, 1.0f);
		nvgFontSize(args.vg, 11);
		nvgFontFaceId(args.vg, font->handle);
		nvgFillColor(args.vg, orange_red_bright);
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		{
			//Vec p = getMarkerStartFromPos(pos);
			//Vec n = getMarkerStartFromPos(pos, RADIUS*0.6);
			drawTextBox(args, p, n, text);
		}
		nvgStroke(args.vg);
		drawTextBoxHalo(args, n);
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

	void drawModelNameHalo(const DrawArgs& args) {
		nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
		//const float width = 30.0;
		//const float height = 10.0;
		const float base_w = b.size.x;
		const float base_h = 10.0;
		const float edge = 7.5;
		const float halo_w = base_w + 2.0 * edge;
		const float halo_h = base_h + 2.0 * edge;

		Vec t_box = b.size.minus(Vec(0,20));
		Vec mid = Vec(base_w/2.0, base_h/2.0);
		Vec center = Vec(0.0, t_box.y).plus(mid);
		Vec bot_l = center.minus(mid.plus(Vec(edge, edge)));

		nvgBeginPath(args.vg);
		NVGpaint p = nvgBoxGradient(args.vg, bot_l.x, bot_l.y, halo_w, halo_h, 12.5, 35.0,
				nvgTransRGBAf(orange_red_bright, halo_brightness),
				nvgTransRGBAf(orange_red_bright,-1.0f));
		nvgRoundedRect(args.vg, bot_l.x, bot_l.y, halo_w , halo_h, 10.0);
		nvgClosePath(args.vg);
		nvgFillPaint(args.vg, p);
		nvgFill(args.vg);
	}

	void drawModelName(const DrawArgs& args, std::shared_ptr<Font> font) {
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

    Vec posToVec(float pos) {
        unsigned int posu = static_cast<unsigned int>(pos);
        // TODO: the 3/4 here is hacky and not correct. not sure what a better solution looks like yet
        float posY = rescale(static_cast<float>(posu / CHANNEL_SIZE_X), 0., 8., -gridscale, gridscale) + y_off;
        float posX = rescale(eucMod(pos,static_cast<float>(CHANNEL_SIZE_X)), 0., 8., -gridscale, gridscale * (3./4.)) + x_off;
        return Vec(posX, posY);
    }

	void drawMarkers(const DrawArgs& args, std::shared_ptr<Font> font) {
		float pos_in_L = module->pos_in_L_param.getValue();
		float pos_in_R = module->pos_in_R_param.getValue();
		float pos_out_L = module->pos_out_L_param.getValue();
		float pos_out_R = module->pos_out_R_param.getValue();

        Vec v_in_L = posToVec(pos_in_L);
        Vec v_in_R = posToVec(pos_in_R);
        Vec v_out_L = posToVec(pos_out_L);
        Vec v_out_R = posToVec(pos_out_R);

		//nvgScissor(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y);

		const char* L_IN = "L_IN";
		const char* R_IN = "R_IN";
		const char* L_OUT = "L_OUT";
		const char* R_OUT = "R_OUT";
		drawMarker(args, v_in_L, L_IN, true, true, font);
		if (module->waveChannel.isModMode()) {
			float amp_in_R = module->waveChannel.amp_in_R;
			float sig_in_R = module->sig_in_R_param.getValue();
			drawModInfo(args, pos_in_R, sig_in_R, amp_in_R);
		} else {
			drawMarker(args, v_in_R, R_IN, false, true, font);
		}		
		drawMarker(args, v_out_L, L_OUT, true, false, font);
		drawMarker(args, v_out_R, R_OUT, false, false, font);

		//nvgResetScissor(args.vg);
	}

	void drawLayer(const DrawArgs& args, int layer) override {
		// get font each draw, fix for Rack in VST context
		std::shared_ptr<Font> font = APP->window->loadFont(fontPath);

		if (!module)
			return;

		halo_brightness = sqrt(settings::haloBrightness);

		if (layer == 1) {
            #ifdef DEBUG_PROBE_WEIGHT
			    history.push_back(module->waveChannel.input_probe_L_window);
            #else
                history.push_back(module->waveChannel.v_a0);
            #endif
			history.pop_front();

			nvgSave(args.vg);
				nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
				drawModelName(args, font);
				drawModelNameHalo(args);
			nvgRestore(args.vg);

			nvgSave(args.vg);
				nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
				Vec center = scaleToBoxByX(Vec(0.f,0.f));
				Vec right = scaleToBoxByX(Vec(1.f,0.f));
				NVGpaint p = nvgRadialGradient(args.vg, center.x, center.y, 0.0, right.x, orange_red_bright, nvgTransRGBAf(orange_red_bright, 0.0));
				//for (auto v : history) {
				//	drawWaveform(args, v, i, p, false);
				//	i++;
				//}

				for (int i = history.size() - 1; i >= 0; i--) {
					drawWaveform(args, history[i], i, p, true);
				}
			nvgRestore(args.vg);

			nvgSave(args.vg);
				drawMarkers(args, font);
			nvgRestore(args.vg);
		} else {
			nvgBeginPath(args.vg);
			nvgRoundedRect(args.vg, b.pos.x, b.pos.y, b.size.x, b.size.y, 10.0);
			nvgFillColor(args.vg, dark_grey);
			nvgFill(args.vg);
			nvgClosePath(args.vg);
		}

		Widget::drawLayer(args, layer);
	}
};