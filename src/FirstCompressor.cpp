#include "rack.hpp"
#include "../src/plugin.hpp"
#include "SimpleLagrange.h"
#include "Lut.hpp"
#include "DspFilters/Filter.h"
#include "DspFilters/Butterworth.h"
#include "Biquad2Fast.hpp"

using namespace rack;

// Define user-specified constants
static const int HISTORY_SIZE = 4; // Total history buffer size
static const int OVERSAMPLE = 16;

struct FirstCompressorModule : Module {
    // Enum identifiers for Params, Inputs, Outputs, Lights
    enum ParamIds {
        PARAM_K,
        PARAM_Q,
        PARAM_S,
        PARAM_T,
        PARAM_C,
        PARAM_R,
        PARAM_B,
        NUM_PARAMS
    };
    enum InputIds {
        INPUT_X, // Audio input
        INPUT_Y,
        NUM_INPUTS
    };
    enum OutputIds {
        OUTPUT_L,
        OUTPUT_R,
        OUTPUT_G,
        NUM_OUTPUTS
    };
    enum LightIds {
        NUM_LIGHTS
    };

    Dsp::SimpleFilter<Dsp::Butterworth::LowPass<8>, 1> aa_filter_L;
    Dsp::SimpleFilter<Dsp::Butterworth::LowPass<8>, 1> aa_filter_R;

    // History buffer to store past input samples
    std::array<double, HISTORY_SIZE> history_x;
    std::array<double, HISTORY_SIZE> history_y;

    float* biquad_out_L[1];
    float* biquad_out_R[1];
    
    float* input_x[1];
    float* input_y[1];

    float* output_x[1];
    float* output_y[1];

    float out_L;
    float out_R;

    // store previous cutoff to see if it changed
    float c_prev = 0.0f;
    
    FastBiquadFilter<BiquadMode::Lowpass> fast_biquad;
    
    float gr_out = 0;
    

    // Constructor: Initialize module and history buffer
    FirstCompressorModule() {
        config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
        configParam(PARAM_K, 0.01, 10.f, 0.5f, "K");
        configParam(PARAM_Q, 0.1f, 100.f, 10.f, "Q");
        configParam(PARAM_S, 1.f, 100.f, 2.f, "S");
        configParam(PARAM_T, -48.f, 12.f, -12.f, "T");
        configParam(PARAM_C, 0.1f, 1000.f, 10.f, "C");
        configParam(PARAM_R, 0.5f, 100.0f, 0.707f, "R");
        configParam(PARAM_B, 0.f, 1.f, 0.f, "B");

        // Configure sample rate or other parameters
        fast_biquad.sampleRate       = 48000.0;
        fast_biquad.freqTarget       = 0.1;
        fast_biquad.resonanceTarget  = 0.707;

        aa_filter_L.setup (8,    // order
            48000.*static_cast<double>(OVERSAMPLE),// sample rate
            16000.);   // cutoff
        aa_filter_R.setup (8,    // order
            48000.*static_cast<double>(OVERSAMPLE),// sample rate
            16000.);   // cutoff

        // Initialize history buffer to zero
        for (int i = 0; i < HISTORY_SIZE; ++i) {
            history_x[i] = 0.0f;
            history_y[i] = 0.0f;
        }

        input_x[0] = new float[OVERSAMPLE];
        input_y[0] = new float[OVERSAMPLE];
        
        output_x[0] = new float[OVERSAMPLE];
        output_y[0] = new float[OVERSAMPLE];
    }
    
    static inline float gainReduction(float x, float k, float S, float T) {
        if (x < 1.e-12f) {
            return 1.;
        }
        const float LN10 = 2.30258509299; // ln(10)
        float lz = 20. * k * fast_log(x) - k * T * LN10;
        float lp;
        if (lz > 0) {
            lp = lz + softplusLUT.get(-lz); 
        } else {
            lp = softplusLUT.get(lz);
        }
        return fast_exp(-lp / (20.f*S*k));
    }
    
    static inline float softAbs(float x, float Q, float B) {
        const float LN2 = 0.69314718056; //ln(2)
        float M = std::abs(x * Q);
        //return (M + std::log1p(std::exp(-2.*M)) - LN2) / Q;
        return B*(1. + tanhLUT.get(Q*x)) + (M + softplusLUT.get(-2.*M) - LN2) / Q;
    }

    void compute(
            std::array<double, HISTORY_SIZE>& l,
            std::array<double, HISTORY_SIZE>& r,
            double Q,
            double k,
            double S,
            double T,
            double B,
            double cutoff,
            double R
            ) {
        out_L = 0.;
        out_R = 0.;
        
        /*
        I'd prefer not to break these sections up,
        but doing so is significantly faster presumably because
        of better cache locality for the LUTs and/or autovectorization
        */
        for (int i = 0; i < OVERSAMPLE; i++) {
            double t = 2. + static_cast<double>(i) / 16.;
            double l_lag = getLagrange(t, l);
            double r_lag = getLagrange(t, r);
            input_x[0][i] = l_lag;
            input_y[0][i] = r_lag;
        }
        
        for (int i = 0; i < OVERSAMPLE; i++) {
            output_x[0][i] = softAbs(input_x[0][i], Q, B);
            output_y[0][i] = softAbs(input_y[0][i], Q, B);
        }
        
        fast_biquad.freqTarget = cutoff / 48000.;
        fast_biquad.resonanceTarget = R;
        // remember to substitute tan LUT
        fast_biquad.processBlock(output_x[0], output_y[0], output_x[0], output_y[0], OVERSAMPLE);
    
        // to show the GR signal
        gr_out = gainReduction(output_x[0][0], k, S, T);
        //gr_out = output_x[0][8];
        
        for (int i = 0; i < OVERSAMPLE; i++) {
            output_x[0][i] = input_x[0][i] * gainReduction(output_x[0][i], k, S, T);
            output_y[0][i] = input_y[0][i] * gainReduction(output_y[0][i], k, S, T);
        }
        
        aa_filter_L.process(OVERSAMPLE, output_x);
        aa_filter_R.process(OVERSAMPLE, output_y);
        
        for (int i = 0; i < OVERSAMPLE; i++) {
            out_L += output_x[0][i];
            out_R += output_y[0][i];
        }
        
        out_L /= static_cast<float>(OVERSAMPLE);
        out_R /= static_cast<float>(OVERSAMPLE);
    }

    void process(const ProcessArgs &args) override {
        // Step 1: Read the current input voltage
        float input_x = inputs[INPUT_X].getVoltage();
        float input_y = inputs[INPUT_Y].getVoltage();

        // Update the history buffer by shifting samples
        // Shift older samples towards the end of the buffer
        for(int i = 0; i < HISTORY_SIZE - 1; ++i) {
            history_x[i] = history_x[i + 1];
            history_y[i] = history_y[i + 1];
        }

        // Insert the new input at the end of the buffer
        history_x[HISTORY_SIZE - 1] = input_x;
        history_y[HISTORY_SIZE - 1] = input_y;

        float k = params[PARAM_K].getValue();
        float q = params[PARAM_Q].getValue();
        float s = params[PARAM_S].getValue();
        float t = params[PARAM_T].getValue();
        float c = params[PARAM_C].getValue();
        float b = params[PARAM_B].getValue();
        float r = params[PARAM_R].getValue();

        compute(history_x, history_y, q, k, s, t, b, c, r);

        // Step 4: Set the output voltage based on y0
        outputs[OUTPUT_L].setVoltage(out_L);
        outputs[OUTPUT_R].setVoltage(out_R);
        outputs[OUTPUT_G].setVoltage(gr_out);
    }
};

//////////////////////////
// Module Widget
//////////////////////////
struct FirstCompressorWidget : ModuleWidget {
    FirstCompressorWidget(FirstCompressorModule *module) {
        setModule(module);

        // Set the panel SVG (Replace with your module's SVG file path)
        setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Norms.svg")));

        // Input Port
        addInput(createInputCentered<PJ301MPort>(Vec(10, 30), module, FirstCompressorModule::INPUT_X));

        // Input Port
        addInput(createInputCentered<PJ301MPort>(Vec(10, 50), module, FirstCompressorModule::INPUT_Y));

        // Output Port
        addOutput(createOutputCentered<PJ301MPort>(Vec(10, 70), module, FirstCompressorModule::OUTPUT_L));
        addOutput(createOutputCentered<PJ301MPort>(Vec(10, 90), module, FirstCompressorModule::OUTPUT_R));
        addOutput(createOutputCentered<PJ301MPort>(Vec(10, 330), module, FirstCompressorModule::OUTPUT_G));

        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 120), module, FirstCompressorModule::PARAM_K));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 140), module, FirstCompressorModule::PARAM_Q));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 160), module, FirstCompressorModule::PARAM_S));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 180), module, FirstCompressorModule::PARAM_T));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 200), module, FirstCompressorModule::PARAM_C));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 220), module, FirstCompressorModule::PARAM_R));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 240), module, FirstCompressorModule::PARAM_B));
    }
};

// Register the module with VCV Rack
Model *modelFirstCompressor = createModel<FirstCompressorModule, FirstCompressorWidget>("FreeSurface-FirstCompressor");
