#include <memory>
#include "plugin.hpp"
#include "osdialog.h"
#include "SvgToVector.hpp"

#define OUTPUT_BUFFER_SIZE 16
#define INPUT_BUFFER_MINIMUM_SIZE 256

struct Vektronix : Module {
	enum ParamIds {
		LOAD_PARAM,
		FREQUENCY_PARAM,
		RAMP_TOGGLE_PARAM,
		STATUS_PARAM,
		RESOLUTION_PARAM,
		HORIZONTAL_POSITION_PARAM,
		VERTICAL_POSITION_PARAM,
		HORIZONTAL_SCALE_PARAM,
		VERTICAL_SCALE_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		VOCT_INPUT,
		RAMP_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		X_POS_OUTPUT,
		Y_POS_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		DISK_LIGHT,
		READY_LIGHT,
		WARN_LIGHT,
		EOC_LIGHT,
		NUM_LIGHTS
	};

    //unique_resampler_stereo_float resampler = std::make_unique<resampler_stereo_float>(OUTPUT_BUFFER_SIZE);
    unique_resampler_stereo_float resampler = 
        std::unique_ptr<resampler_stereo_float>(
            new resampler_stereo_float(OUTPUT_BUFFER_SIZE, INPUT_BUFFER_MINIMUM_SIZE)
        );

    dsp::ClockDivider lightDivider;

    dsp::ClockDivider rampDivider;

    std::string lastLoadedPath;

    bool loaded = false;
    bool warn = false;
    float sampleTime;

	Vektronix() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(LOAD_PARAM, 0.f, 1.f, 0.f, "Load SVG From Disk");
		configParam(FREQUENCY_PARAM, -8.0f, 8.0f, 0.f, "Frequency shift (V/Oct)");
		configParam(RAMP_TOGGLE_PARAM, 0.f, 1.f, 0.f, "");
		configParam(STATUS_PARAM, 0.f, 1.f, 0.f, "");
		configParam(RESOLUTION_PARAM, 0.f, 1.f, 0.f, "");
		configParam(HORIZONTAL_POSITION_PARAM, -5.f, 5.f, 0.f, "Horizontal position");
		configParam(VERTICAL_POSITION_PARAM, -5.f, 5.f, 0.f, "Vertical position");
		configParam(HORIZONTAL_SCALE_PARAM, -2.f, 2.f, 1.f, "Horizontal scale");
		configParam(VERTICAL_SCALE_PARAM, -2.f, 2.f, 1.f, "Vertical scale");
        lightDivider.setDivision(16);

	}

    float getRatio(float pitch) {
		//return std::max(0.125f, dsp::approxExp2_taylor5<float>(pitch + 30) / 1073741824);
        return std::max(0.0625f, std::exp2(-pitch));
	}

    float ramp() {
        return static_cast<float>(lightDivider.getClock()) / static_cast<float>(lightDivider.getDivision());
    }

    void setLight(LightIds id, float val) {
        lights[id].setSmoothBrightness(val, sampleTime * lightDivider.getDivision());
    }

	void process(const ProcessArgs& args) override {
        sampleTime = args.sampleTime;
        bool EOC = rampDivider.process();
        float x_val, y_val;
        if (loaded && resampler->ready()) {
            float voct_in = inputs[VOCT_INPUT].getVoltage(0);
            float ratio = getRatio(math::clamp(params[FREQUENCY_PARAM].getValue() + voct_in,-8.0,8.0));
            resampler->setResampleRatio(ratio);
            StereoSample s = resampler->shiftOutput();
            x_val = s.x;
            y_val = s.y;
            warn = s.bufferEmpty;

            float h_pos = params[HORIZONTAL_POSITION_PARAM].getValue();
            float h_scl = params[HORIZONTAL_SCALE_PARAM].getValue();
            float v_pos = params[VERTICAL_POSITION_PARAM].getValue();
            float v_scl = params[VERTICAL_SCALE_PARAM].getValue();
            outputs[X_POS_OUTPUT].setChannels(1);
            outputs[Y_POS_OUTPUT].setChannels(1);
            outputs[X_POS_OUTPUT].setVoltage(x_val * h_scl + h_pos, 0);
            outputs[Y_POS_OUTPUT].setVoltage(y_val * v_scl + v_pos, 0);
        } else {
            outputs[X_POS_OUTPUT].setChannels(1);
            outputs[Y_POS_OUTPUT].setChannels(1);
            outputs[X_POS_OUTPUT].setVoltage(0.0, 0);
            outputs[Y_POS_OUTPUT].setVoltage(0.0, 0); 
        }

        if (lightDivider.process()) {
            setLight(DISK_LIGHT, loaded ? 1.0 : 0.0);
            setLight(READY_LIGHT, loaded ? 1.0 : 0.0);
            setLight(WARN_LIGHT, warn ? 1.0 : 0.0);
            setLight(EOC_LIGHT, EOC ? 1.0 : 0.0);
        }

	}

    // file selection dialog, based on PLAYERItem in cf
    // https://github.com/cfoulc/cf/blob/master/src/PLAYER.cpp
    void getSvgPathDialog() {
        std::string dir = lastLoadedPath.empty() ? asset::user("") : rack::string::directory(lastLoadedPath);
        osdialog_filters* filters = osdialog_filters_parse(".svg files:svg");
        char *path = osdialog_file(OSDIALOG_OPEN, dir.c_str(), NULL, filters);
        if(path) {
            loadSvg(path);
            lastLoadedPath = path;
            free(path);
        }
        osdialog_filters_free(filters);
    }

    void loadSvg(std::string path) {
        SvgToVector* loader = new SvgToVector(10.0, 10.0, 1.0);
        try {
            resampler->reset();
            loader->loadSvg(path.c_str(), resampler);
            resampler->finalize();
            rampDivider.setDivision(resampler->size());
            rampDivider.reset();
            loaded = true;
            warn = false;
        } catch (const std::runtime_error& err) {
            DEBUG("runtime error: %d", errno);
            loaded = false;
            warn = true;
        } catch (...) {
            DEBUG("unknown error: %d", errno);
            loaded = false;
            warn = true;
        }

    }
};

struct VektronixLoadButton : VektronixToggle {
    Vektronix* module;
    VektronixLoadButton() {
        this->momentary = true;
    }

    void onButton(const event::Button& e) override {
		//ParamWidget::onButton(e);

        e.stopPropagating();
		if (!module) {
			return;
        }

		if (e.action == GLFW_PRESS && (e.button == GLFW_MOUSE_BUTTON_LEFT || e.button == GLFW_MOUSE_BUTTON_RIGHT)) {
            module->getSvgPathDialog();
			e.consume(this);
		}
	}
};


struct VektronixWidget : ModuleWidget {
	VektronixWidget(Vektronix* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Vektronix.svg")));

        VektronixLoadButton* loadButton = createParamCentered<VektronixLoadButton>(mm2px(Vec(38.5, 28.679)), module, Vektronix::LOAD_PARAM);
        loadButton->module = module;
		addParam(loadButton);
        addParam(createParamCentered<VektronixToggle>(mm2px(Vec(38.5, 37.391)), module, Vektronix::RAMP_TOGGLE_PARAM));
		addParam(createParamCentered<VektronixToggle>(mm2px(Vec(38.5, 46.104)), module, Vektronix::STATUS_PARAM));

		addParam(createParamCentered<VektronixBigKnob>(mm2px(Vec(14.0, 32.574)), module, Vektronix::FREQUENCY_PARAM));
		addParam(createParamCentered<VektronixSmallKnob>(mm2px(Vec(14.0, 50.113)), module, Vektronix::RESOLUTION_PARAM));
		addParam(createParamCentered<VektronixSmallKnob>(mm2px(Vec(36.816, 69.95)), module, Vektronix::HORIZONTAL_POSITION_PARAM));
		addParam(createParamCentered<VektronixSmallKnob>(mm2px(Vec(14.0, 69.95)), module, Vektronix::VERTICAL_POSITION_PARAM));
		addParam(createParamCentered<VektronixBigKnob>(mm2px(Vec(36.816, 90.75)), module, Vektronix::HORIZONTAL_SCALE_PARAM));
		addParam(createParamCentered<VektronixBigKnob>(mm2px(Vec(14.0, 90.75)), module, Vektronix::VERTICAL_SCALE_PARAM));

		addInput(createInputCentered<VektronixPort>(mm2px(Vec(30.744, 120.0)), module, Vektronix::VOCT_INPUT));
		addInput(createInputCentered<VektronixPort>(mm2px(Vec(41.93, 120.0)), module, Vektronix::RAMP_INPUT));
		addOutput(createOutputCentered<VektronixPort>(mm2px(Vec(8.177, 120.0)), module, Vektronix::X_POS_OUTPUT));
		addOutput(createOutputCentered<VektronixPort>(mm2px(Vec(19.428, 120.0)), module, Vektronix::Y_POS_OUTPUT));

		addChild(createLightCentered<VektronixDiskLight<GreenLight>>(mm2px(Vec(10.795, 16.276)), module, Vektronix::DISK_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(38.156, 51.063)), module, Vektronix::READY_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(38.156, 52.978)), module, Vektronix::WARN_LIGHT));
		addChild(createLightCentered<TinyLight<RedLight>>(mm2px(Vec(38.156, 54.866)), module, Vektronix::EOC_LIGHT));
	}
};

Model* modelVektronix = createModel<Vektronix, VektronixWidget>("FreeSurface-Vektronix");

#undef OUTPUT_BUFFER_SIZE