// LG57: 4x local/guard oversampled waveshaper whose nonlinearity is a
// polynomial connector between two exterior lines (src/dsp/connector.hpp):
//
//   x < -1 : left line   value_L + slope_L * (x + 1)
//   |x|<=1 : the unique minimum-degree Hermite polynomial matching the
//            lines through r = floor((D - 1)/2) derivatives (degree 2r+1)
//   x > +1 : right line  value_R + slope_R * (x - 1)
//
// Knobs: the two line values and slopes, and the degree cap D (the
// maximum harmonic degree the interior polynomial may generate).  The
// default (-1, 0, +1, 0, D = 7) is the study's d7 odd smoothstep.  Knobs
// are not CV-modulatable: the connector is re-solved (a <= 16x16 system)
// only when a knob moves, at most every 32 samples.
//
// DC removal: a one-pole DC blocker (5 Hz) on the output, toggled by the
// DC button (red light when active).
//
// FAST/EXACT: the FAST core is a 512-segment cubic-Hermite LUT of the
// connector (dsp/lut_connector.hpp; alias floors identical to the exact
// evaluator, ~43 ns/sample flat in degree), rebuilt a few knots per
// sample after a knob edit with the previous table active meanwhile.
// EXACT evaluates the factored bases directly.  One button toggles; a
// red light marks the selected mode.
//
// Oversampling scaffold (generated/lg57_config.hpp, the study's LG57
// lowering): 199-tap least-squares FIR to 2x, exact-structure 14-tap
// midpoint to 4x, core, 110 dB halfband to 2x, 140 dB guard halfband to
// 1x.  Rack audio (+-5 V) maps to the connector's +-1 domain; the output
// is the connector value * 5 V (the exterior lines are not clamped).
// Designed at 48 kHz (filters scale with the engine rate); group delay
// 78.75 samples.
#include "plugin.hpp"
#include "dsp/connector.hpp"
#include "dsp/lut_connector.hpp"
#include "dsp/shaper4x.hpp"
#include "generated/lg57_config.hpp"
#include "FixedsysLabels.hpp"
#include "ConnectorDisplay.hpp"

struct LG57 : Module {
    // the first CONNECTOR_PARAMS parameters define the connector
    enum ParamId { LEFT_VALUE_PARAM, LEFT_SLOPE_PARAM, RIGHT_VALUE_PARAM,
                   RIGHT_SLOPE_PARAM, DEGREE_PARAM, DC_PARAM, MODE_PARAM,
                   PARAMS_LEN };
    static constexpr int CONNECTOR_PARAMS = DEGREE_PARAM + 1;
    enum InputId { IN_INPUT, INPUTS_LEN };
    enum OutputId { OUT_OUTPUT, OUTPUTS_LEN };
    enum LightId { DC_LIGHT, FAST_LIGHT, EXACT_LIGHT, LIGHTS_LEN };

    static constexpr float kInputScale = 1.f / 5.f;
    static constexpr float kOutputScale = 5.f;
    static constexpr int kParamCheckInterval = 32;

    polydist::Shaper4x<aadsp::generated::lg57_config> shaper{1};
    polydist::LineConnector connector;
    polydist::LutCore lut;
    bool lutDirty = true;
    static constexpr int kLutKnotsPerSample = 4;
    float cached[CONNECTOR_PARAMS] = {};
    int paramCounter = 0;

    // DC blocker y[n] = x[n] - x[n-1] + R y[n-1], R from a 5 Hz corner
    static constexpr float kDcCornerHz = 5.f;
    float dcR = 0.999f, dcX1 = 0.f, dcY1 = 0.f;

    LG57() {
        config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
        configParam(LEFT_VALUE_PARAM, -2.f, 2.f, -1.f, "Left line value");
        configParam(LEFT_SLOPE_PARAM, -5.f, 5.f, 0.f, "Left line slope");
        configParam(RIGHT_VALUE_PARAM, -2.f, 2.f, 1.f, "Right line value");
        configParam(RIGHT_SLOPE_PARAM, -5.f, 5.f, 0.f, "Right line slope");
        configParam(DEGREE_PARAM, 1.f, (float)polydist::kMaxDegreeCap, 7.f,
                    "Maximum harmonic degree");
        getParamQuantity(DEGREE_PARAM)->snapEnabled = true;
        configSwitch(DC_PARAM, 0.f, 1.f, 0.f, "DC removal", {"Off", "On"});
        configSwitch(MODE_PARAM, 0.f, 1.f, 1.f, "Core", {"Exact", "Fast"});
        configInput(IN_INPUT, "Audio");
        configOutput(OUT_OUTPUT, "Audio");
        configBypass(IN_INPUT, OUT_OUTPUT);
        updateConnector(true);
        updateDcCoefficient(APP->engine->getSampleRate());
    }

    void updateConnector(bool force) {
        bool changed = force;
        for (int i = 0; i < CONNECTOR_PARAMS; ++i) {
            const float v = params[i].getValue();
            if (v != cached[i]) { cached[i] = v; changed = true; }
        }
        if (!changed) return;
        polydist::connector_between_lines(
            cached[LEFT_VALUE_PARAM], cached[LEFT_SLOPE_PARAM],
            cached[RIGHT_VALUE_PARAM], cached[RIGHT_SLOPE_PARAM],
            (int)std::round(cached[DEGREE_PARAM]), connector);
        lutDirty = true;
    }

    void updateDcCoefficient(float sampleRate) {
        dcR = 1.f - 2.f * M_PI * kDcCornerHz / sampleRate;
    }

    void onReset() override {
        shaper.reset();
        dcX1 = dcY1 = 0.f;
        updateConnector(true);
    }
    void onSampleRateChange(const SampleRateChangeEvent& e) override {
        shaper.reset();
        dcX1 = dcY1 = 0.f;
        updateDcCoefficient(e.sampleRate);
    }

    void process(const ProcessArgs& args) override {
        if (++paramCounter >= kParamCheckInterval) {
            paramCounter = 0;
            updateConnector(false);
        }
        const bool fast = params[MODE_PARAM].getValue() > 0.5f;
        if (fast) {
            if (lutDirty && !lut.building()) {
                lut.beginBuild(connector);
                lutDirty = false;
            }
            if (lut.building()) lut.step(kLutKnotsPerSample);
        }
        lights[FAST_LIGHT].setBrightness(fast ? 1.f : 0.f);
        lights[EXACT_LIGHT].setBrightness(fast ? 0.f : 1.f);

        float x = inputs[IN_INPUT].getVoltage() * kInputScale;
        float y = 0.f;
        if (fast && lut.ready())
            shaper.process_block(&x, &y, 1, lut.table());
        else
            shaper.process_block(&x, &y, 1, connector);
        float out = y * kOutputScale;
        const bool dc = params[DC_PARAM].getValue() > 0.5f;
        if (dc) {
            const float filtered = out - dcX1 + dcR * dcY1;
            dcX1 = out;
            dcY1 = filtered;
            out = filtered;
        } else {
            dcX1 = out;
            dcY1 = 0.f;
        }
        outputs[OUT_OUTPUT].setVoltage(out);
        lights[DC_LIGHT].setBrightness(dc ? 1.f : 0.f);
    }
};

struct LG57Widget : ModuleWidget {
    LG57Widget(LG57* module) {
        setModule(module);
        setPanel(createPanel(asset::plugin(pluginInstance, "res/LG57.svg")));

        addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
        addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
        addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
        addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

        // labels: Fixedsys at the display size, one baseline per row
        FixedsysLabels* labels = new FixedsysLabels;
        labels->box.pos = Vec(0, 0);
        labels->box.size = box.size;
        labels->add("LG57", 30.48f, 6.9f);
        labels->add("L VAL", 9.5f, 69.0f);
        labels->add("L SLP", 23.5f, 69.0f);
        labels->add("R VAL", 37.5f, 69.0f);
        labels->add("R SLP", 51.5f, 69.0f);
        labels->add("DC", 9.5f, 86.5f);
        labels->add("DEGREE", 30.48f, 86.5f);
        labels->add("FAST", 48.5f, 86.5f);
        labels->add("EXACT", 48.5f, 103.0f);
        labels->add("IN", 12.0f, 109.5f);
        labels->add("OUT", 49.0f, 109.5f);
        addChild(labels);

        ConnectorDisplay<LG57>* display = new ConnectorDisplay<LG57>();
        display->module = module;
        display->box.pos = mm2px(Vec(3.0, 11.0));
        display->box.size = mm2px(Vec(54.96, 52.0));
        addChild(display);

        addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(9.5, 76.0)), module, LG57::LEFT_VALUE_PARAM));
        addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(23.5, 76.0)), module, LG57::LEFT_SLOPE_PARAM));
        addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(37.5, 76.0)), module, LG57::RIGHT_VALUE_PARAM));
        addParam(createParamCentered<VektronixSmallKnobDark>(mm2px(Vec(51.5, 76.0)), module, LG57::RIGHT_SLOPE_PARAM));
        addParam(createParamCentered<VektronixBigKnobDark>(mm2px(Vec(30.48, 96.5)), module, LG57::DEGREE_PARAM));

        addParam(createParamCentered<VektronixRoundToggleDark>(mm2px(Vec(9.5, 93.5)), module, LG57::DC_PARAM));
        addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(9.5, 99.0)), module, LG57::DC_LIGHT));

        addParam(createParamCentered<VektronixRoundToggleDark>(mm2px(Vec(51.5, 93.5)), module, LG57::MODE_PARAM));
        addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.8, 85.4)), module, LG57::FAST_LIGHT));
        addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(55.8, 101.9)), module, LG57::EXACT_LIGHT));

        addInput(createInputCentered<VektronixPortBorderlessDark>(mm2px(Vec(12.0, 116.0)), module, LG57::IN_INPUT));
        addOutput(createOutputCentered<VektronixPortBorderlessDark>(mm2px(Vec(49.0, 116.0)), module, LG57::OUT_OUTPUT));
    }
};

Model* modelLG57 = createModel<LG57, LG57Widget>("FreeSurface-LG57");
