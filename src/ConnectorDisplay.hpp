#pragma once
// Amber-plasma display of the connector curve, in the idiom of the
// WaterTable display in FreeSurface (same palette, additive NVG_LIGHTER
// glow layering, XOR'd text on solid orange boxes, Fixedsys Excelsior):
//   - the curve over [-1.6, 1.6] so the exterior lines show past the joins
//   - dimmer dashed orange axes and gradations (+-1 joins, +-0.5)
//   - a bottom bar naming the selected degree cap / connector degree /
//     smoothness (the WaterTable model-name bar)
//   - "L_END" / "R_END" markers with leader lines that follow the two
//     line endpoints (the WaterTable position markers)
#include "plugin.hpp"
#include "dsp/connector.hpp"

template <class TModule>
struct ConnectorDisplay : TransparentWidget {
    TModule* module = nullptr;

    const NVGcolor orange_red_bright = nvgRGBA(0xf5, 0x39, 0x0a, 0xff);
    const NVGcolor orange_red = nvgRGBA(0xd0, 0x28, 0x0a, 0xff);
    const NVGcolor ember_orange = nvgRGBA(0xff, 0xcf, 0x3f, 0xff);
    const NVGcolor hot_white = nvgRGBA(0xff, 0xff, 0xeb, 0xff);
    const NVGcolor dark_grey = nvgRGBA(0x10, 0x10, 0x10, 0xff);

    static constexpr float X_RANGE = 1.6f;      // plotted input range
    static constexpr float BAR_H = 10.f;
    static constexpr int CURVE_SAMPLES = 192;

    polydist::LineConnector conn;
    float cached[TModule::CONNECTOR_PARAMS];
    bool have = false;
    float yrange = 1.6f;

    ConnectorDisplay() {
        polydist::connector_between_lines(-1, 0, 1, 0, 7, conn);
        for (float& c : cached) c = NAN;
    }

    // Re-solve from the knobs on the UI thread (the audio thread keeps
    // its own copy); the browser preview (no module) shows the default.
    void updateConnector() {
        if (!module) return;
        bool changed = !have;
        for (int i = 0; i < TModule::CONNECTOR_PARAMS; ++i) {
            float v = module->params[i].getValue();
            if (v != cached[i]) { cached[i] = v; changed = true; }
        }
        if (!changed) return;
        have = true;
        polydist::connector_between_lines(
            cached[TModule::LEFT_VALUE_PARAM], cached[TModule::LEFT_SLOPE_PARAM],
            cached[TModule::RIGHT_VALUE_PARAM], cached[TModule::RIGHT_SLOPE_PARAM],
            (int)std::round(cached[TModule::DEGREE_PARAM]), conn);
        // symmetric vertical range following the curve continuously (a
        // little headroom above the extreme value), never below the joins
        double m = 1.0;
        for (int i = 0; i <= CURVE_SAMPLES; ++i) {
            double x = -X_RANGE + 2.0 * X_RANGE * i / CURVE_SAMPLES;
            m = std::max(m, std::fabs(conn(x)));
        }
        yrange = (float)std::max(1.6, m * 1.12);
    }

    Rect plotRect() const {
        return Rect(Vec(0, 0), Vec(box.size.x, box.size.y - BAR_H));
    }
    Vec toScreen(const Rect& p, double x, double y) const {
        return Vec(rescale((float)x, -X_RANGE, X_RANGE, p.pos.x, p.pos.x + p.size.x),
                   rescale((float)y, yrange, -yrange, p.pos.y, p.pos.y + p.size.y));
    }

    // nanovg has no dash pattern; draw segments
    static void dashedLine(NVGcontext* vg, Vec a, Vec b, float dash, float gap) {
        Vec d = b.minus(a);
        float len = d.norm();
        if (len <= 0.f) return;
        Vec u = d.div(len);
        nvgBeginPath(vg);
        for (float t = 0.f; t < len; t += dash + gap) {
            Vec s = a.plus(u.mult(t));
            Vec e = a.plus(u.mult(std::min(len, t + dash)));
            nvgMoveTo(vg, s.x, s.y);
            nvgLineTo(vg, e.x, e.y);
        }
        nvgStroke(vg);
    }

    void drawGrid(const DrawArgs& args, const Rect& p) {
        nvgSave(args.vg);
        nvgScissor(args.vg, p.pos.x, p.pos.y, p.size.x, p.size.y);
        nvgLineCap(args.vg, NVG_BUTT);
        // minor gradations at +-0.5 (and further half-units within range)
        nvgStrokeWidth(args.vg, 0.8f);
        nvgStrokeColor(args.vg, nvgTransRGBAf(orange_red, 0.22f));
        for (float g = 0.5f; g < X_RANGE; g += 1.0f)
            for (float s : {-1.f, 1.f}) {
                dashedLine(args.vg, toScreen(p, s * g, -yrange), toScreen(p, s * g, yrange), 2.f, 4.f);
            }
        for (float g = 0.5f; g < yrange; g += 1.0f)
            for (float s : {-1.f, 1.f})
                dashedLine(args.vg, toScreen(p, -X_RANGE, s * g), toScreen(p, X_RANGE, s * g), 2.f, 4.f);
        // the joins at +-1 and the unit levels
        nvgStrokeWidth(args.vg, 1.0f);
        nvgStrokeColor(args.vg, nvgTransRGBAf(orange_red, 0.4f));
        for (float s : {-1.f, 1.f}) {
            dashedLine(args.vg, toScreen(p, s, -yrange), toScreen(p, s, yrange), 4.f, 3.f);
            dashedLine(args.vg, toScreen(p, -X_RANGE, s), toScreen(p, X_RANGE, s), 4.f, 3.f);
        }
        // axes
        nvgStrokeColor(args.vg, nvgTransRGBAf(orange_red, 0.55f));
        dashedLine(args.vg, toScreen(p, 0, -yrange), toScreen(p, 0, yrange), 6.f, 3.f);
        dashedLine(args.vg, toScreen(p, -X_RANGE, 0), toScreen(p, X_RANGE, 0), 6.f, 3.f);
        nvgResetScissor(args.vg);
        nvgRestore(args.vg);
    }

    void tracePath(NVGcontext* vg, const Rect& p, double x0, double x1) {
        nvgBeginPath(vg);
        const int n = std::max(8, (int)(CURVE_SAMPLES * (x1 - x0) / (2 * X_RANGE)));
        for (int i = 0; i <= n; ++i) {
            double x = x0 + (x1 - x0) * i / n;
            Vec s = toScreen(p, x, conn(x));
            if (i == 0) nvgMoveTo(vg, s.x, s.y); else nvgLineTo(vg, s.x, s.y);
        }
    }

    void glowStroke(NVGcontext* vg, const Rect& p, double x0, double x1, NVGcolor core, float coreW) {
        // plasma: wide faint halo, mid glow, bright core (additive)
        nvgLineCap(vg, NVG_ROUND);
        nvgLineJoin(vg, NVG_ROUND);
        tracePath(vg, p, x0, x1);
        nvgStrokeWidth(vg, coreW * 5.f);
        nvgStrokeColor(vg, nvgTransRGBAf(orange_red, 0.16f));
        nvgStroke(vg);
        tracePath(vg, p, x0, x1);
        nvgStrokeWidth(vg, coreW * 2.2f);
        nvgStrokeColor(vg, nvgTransRGBAf(nvgLerpRGBA(orange_red, ember_orange, 0.5f), 0.45f));
        nvgStroke(vg);
        tracePath(vg, p, x0, x1);
        nvgStrokeWidth(vg, coreW);
        nvgStrokeColor(vg, core);
        nvgStroke(vg);
    }

    void drawCurve(const DrawArgs& args, const Rect& p) {
        nvgSave(args.vg);
        nvgScissor(args.vg, p.pos.x, p.pos.y, p.size.x, p.size.y);
        nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
        // exterior lines: dimmer ember; interior polynomial: hot core
        glowStroke(args.vg, p, -X_RANGE, conn.xL, nvgTransRGBAf(ember_orange, 0.85f), 1.3f);
        glowStroke(args.vg, p, conn.xR, X_RANGE, nvgTransRGBAf(ember_orange, 0.85f), 1.3f);
        glowStroke(args.vg, p, conn.xL, conn.xR, nvgLerpRGBA(ember_orange, hot_white, 0.55f), 1.6f);
        // joins
        for (double x : {conn.xL, conn.xR}) {
            Vec s = toScreen(p, x, conn(x));
            nvgBeginPath(args.vg);
            nvgCircle(args.vg, s.x, s.y, 4.5f);
            nvgFillColor(args.vg, nvgTransRGBAf(orange_red, 0.35f));
            nvgFill(args.vg);
            nvgBeginPath(args.vg);
            nvgCircle(args.vg, s.x, s.y, 2.0f);
            nvgFillColor(args.vg, hot_white);
            nvgFill(args.vg);
        }
        nvgResetScissor(args.vg);
        nvgRestore(args.vg);
    }

    // WaterTable's drawTextBox: solid box, leader line, XOR'd dark text
    void drawTextBox(const DrawArgs& args, Vec anchor, Vec center, const char* text) {
        const float width = 30.f, height = 10.f;
        Vec half(width / 2.f, height / 2.f);
        Vec top_l = center.plus(Vec(-half.x, -half.y)), bot_r = center.plus(half);
        // leader from the anchor to the nearest point on the box edge
        Vec c = Vec(clamp(anchor.x, top_l.x, bot_r.x), clamp(anchor.y, top_l.y, bot_r.y));
        nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
        nvgStrokeColor(args.vg, orange_red_bright);
        nvgStrokeWidth(args.vg, 1.0f);
        nvgLineCap(args.vg, NVG_SQUARE);
        nvgBeginPath(args.vg);
        nvgMoveTo(args.vg, anchor.x, anchor.y);
        nvgLineTo(args.vg, c.x, c.y);
        nvgStroke(args.vg);
        nvgFillColor(args.vg, orange_red_bright);
        nvgBeginPath(args.vg);
        nvgRect(args.vg, top_l.x, top_l.y, width, height);
        nvgFill(args.vg);
        nvgGlobalCompositeOperation(args.vg, NVG_XOR);
        nvgFillColor(args.vg, dark_grey);
        nvgFontSize(args.vg, 11);
        nvgTextAlign(args.vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
        nvgText(args.vg, top_l.x + 2.f, bot_r.y - 2.f, text, NULL);
        nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
    }

    void drawMarkers(const DrawArgs& args, const Rect& p) {
        const float ox = 26.f, oy = 15.f, m = 4.f;
        auto place = [&](Vec anchor, float dx, float dy) {
            Vec c = anchor.plus(Vec(dx, dy));
            c.x = clamp(c.x, p.pos.x + 15.f + m, p.pos.x + p.size.x - 15.f - m);
            c.y = clamp(c.y, p.pos.y + 5.f + m, p.pos.y + p.size.y - 5.f - m);
            return c;
        };
        // left endpoint: box on the side away from where the exterior line runs
        Vec aL = toScreen(p, conn.xL, conn.left_value);
        drawTextBox(args, aL, place(aL, -ox, conn.left_slope >= 0 ? -oy : oy), "L_END");
        Vec aR = toScreen(p, conn.xR, conn.right_value);
        drawTextBox(args, aR, place(aR, ox, conn.right_slope <= 0 ? -oy : oy), "R_END");
    }

    void drawBar(const DrawArgs& args) {
        const float y = box.size.y - BAR_H;
        nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
        nvgFillColor(args.vg, orange_red_bright);
        nvgBeginPath(args.vg);
        nvgRect(args.vg, 0.f, y, box.size.x, BAR_H);
        nvgFill(args.vg);
        char text[64];
        const int deg = conn.interior_degree();
        std::snprintf(text, sizeof(text), "MAX DEG %d  >  DEG %d  C%d",
                      conn.degree_cap, deg, conn.smoothness_order);
        nvgGlobalCompositeOperation(args.vg, NVG_XOR);
        nvgFillColor(args.vg, hot_white);
        nvgFontSize(args.vg, 11);
        nvgTextAlign(args.vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
        nvgText(args.vg, box.size.x / 2.f, y + BAR_H - 2.f, text, NULL);
        nvgGlobalCompositeOperation(args.vg, NVG_LIGHTER);
    }

    void draw(const DrawArgs& args) override {
        updateConnector();
        std::shared_ptr<Font> font = APP->window->loadFont(
            asset::plugin(pluginInstance, "res/fixedsys-excelsior-301.ttf"));
        if (!font) return;
        nvgFontFaceId(args.vg, font->handle);
        const Rect p = plotRect();

        nvgBeginPath(args.vg);
        nvgRoundedRect(args.vg, 0, 0, box.size.x, box.size.y, 10.f);
        nvgFillColor(args.vg, dark_grey);
        nvgFill(args.vg);

        nvgSave(args.vg);
        nvgScissor(args.vg, 0, 0, box.size.x, box.size.y);
        drawGrid(args, p);
        drawCurve(args, p);
        nvgSave(args.vg);
        drawMarkers(args, p);
        nvgRestore(args.vg);
        nvgSave(args.vg);
        drawBar(args);
        nvgRestore(args.vg);
        nvgResetScissor(args.vg);
        nvgRestore(args.vg);
    }
};
