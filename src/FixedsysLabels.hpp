#pragma once
// Panel labels in Fixedsys Excelsior at the display's size (11 px), each
// row sharing one baseline.  Positions in mm.  (Ported back from the
// Polydist plugin's FreeSurfaceComponents.hpp.)
#include "plugin.hpp"

struct FixedsysLabels : widget::TransparentWidget {
    struct Label { std::string text; float x_mm; float baseline_mm; int align; };
    std::vector<Label> labels;
    NVGcolor ink = nvgRGB(0xe3, 0xe2, 0xdb);
    float fontSize = 11.f;

    void add(const std::string& text, float x_mm, float baseline_mm, int align = NVG_ALIGN_CENTER) {
        labels.push_back({text, x_mm, baseline_mm, align});
    }
    void draw(const DrawArgs& args) override {
        std::shared_ptr<window::Font> font = APP->window->loadFont(
            asset::plugin(pluginInstance, "res/fixedsys-excelsior-301.ttf"));
        if (!font) return;
        nvgFontFaceId(args.vg, font->handle);
        nvgFontSize(args.vg, fontSize);
        nvgFillColor(args.vg, ink);
        for (const Label& l : labels) {
            nvgTextAlign(args.vg, l.align | NVG_ALIGN_BASELINE);
            math::Vec p = mm2px(math::Vec(l.x_mm, l.baseline_mm));
            nvgText(args.vg, p.x, p.y, l.text.c_str(), NULL);
        }
    }
};
