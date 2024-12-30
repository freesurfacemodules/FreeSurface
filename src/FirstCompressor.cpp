#include "rack.hpp"
#include "../src/plugin.hpp"
#include "SimpleLagrange.h"
#include "Lut.hpp"
#include "DspFilters/Filter.h"
#include "DspFilters/Butterworth.h"
#include "DspFilters/RBJ.h"
#include "DspFilters/SmoothedFilter.h"

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
        NUM_OUTPUTS
    };
    enum LightIds {
        NUM_LIGHTS
    };

    dsp::TBiquadFilter<double> biquad_input_L;
    dsp::TBiquadFilter<double> biquad_input_R;

    Dsp::Filter* biquad_L;
    Dsp::Filter* biquad_R;

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
    
    float slew_prev_L = 0.f;
    float slew_prev_R = 0.f;

    const float biquad_cutoff = 0.25f;
    const float biquad_Q = 0.25f;
    const float biquad_gain = 1.0f;

    // store previous cutoff to see if it changed
    float c_prev = 0.0f;

    Dsp::Params biquad_params;


    // Constructor: Initialize module and history buffer
    FirstCompressorModule() {
        config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
        configParam(PARAM_K, 0.01, 10.f, 0.5f, "K");
        configParam(PARAM_Q, 0.1f, 100.f, 10.f, "Q");
        configParam(PARAM_S, 1.f, 100.f, 2.f, "S");
        configParam(PARAM_T, -48.f, 12.f, -12.f, "T");
        configParam(PARAM_C, 2.f, 10000.f, 100.f, "C");
        configParam(PARAM_B, 0.f, 1.f, 0.f, "B");


        biquad_input_L.setParameters(dsp::TBiquadFilter<double>::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);
        biquad_input_R.setParameters(dsp::TBiquadFilter<double>::Type::LOWPASS,  biquad_cutoff, biquad_Q, biquad_gain);

        biquad_L = new Dsp::SmoothedFilterDesign
                <Dsp::RBJ::Design::LowPass, 1> (1024*OVERSAMPLE);
        biquad_R = new Dsp::SmoothedFilterDesign
                <Dsp::RBJ::Design::LowPass, 1> (1024*OVERSAMPLE);

        biquad_params[0] = 48000*static_cast<double>(OVERSAMPLE); // sample rate
        biquad_params[1] = 100; // cutoff frequency
        biquad_params[2] = 0.5; // Q
        biquad_L->setParams (biquad_params);
        biquad_R->setParams (biquad_params);

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

        // format required by library
        biquad_out_L[0] = new float[1];
        biquad_out_R[0] = new float[1];
    }

    // https://www.desmos.com/calculator/iki5uxzlfo
    // gain reduction function converted to log-sum-exp form to avoid overflow
    // x should be >= 0
    // S should be 1 <= S <= 100
    // T should be -48 <= T <= 12
    // k should be 0.1 <= k <= 10
    /*static double gainReduction(double x, double k, double S, double T) {
        if (x < 1.e-12) {
            return 1.;
        }
        const double LN10 = std::log(10.);
        double lz = 20. * k * std::log(x) - k * T * LN10;
        double lp;
        if (lz > 0) {
            lp = lz + std::log1p(std::exp(-lz));
        } else {
            lp = std::log1p(std::exp(lz));
        }
        return std::exp(-lp / (20.*S*k));
    }*/
    
    /*static double gainReduction(double x, double k, double S, double T) {
        if (x < 1.e-12) {
            return 1.;
        }
        const double LN10 = 2.30258509299; // ln(10)
        //double lz = 20. * k * std::log(x) - k * T * LN10;
        double lz = 20. * k * fast_log(x) - k * T * LN10;
        double lp;
        if (lz > 0) {
            //lp = lz + std::log1p(std::exp(-lz));
            lp = lz + softplusLUT.get(-lz); 
        } else {
            //lp = std::log1p(std::exp(lz));
            lp = softplusLUT.get(lz);
        }
        //return std::exp(-lp / (20.*S*k));
        return fast_exp(-lp / (20.*S*k));
    }*/
    
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

    // Q should vary between 1 and 100
    // B should vary between 0 and 1
    // this is ln(cosh(x*Q))/Q but modified to avoid overflow
    /*static double softAbs(double x, double Q, double B) {
        const double LN2 = log(2.);
        double M = std::abs(x * Q);
        //return (M + std::log1p(std::exp(-2.*M)) - LN2) / Q;
        return B*(1. + std::tanh(Q*x)) + (M + std::log1p(std::exp(-2.*M)) - LN2) / Q;
    }*/
    
    /*static inline double softAbs(double x, double Q, double B) {
        const double LN2 = 0.69314718056; //ln(2)
        double M = std::abs(x * Q);
        //return (M + std::log1p(std::exp(-2.*M)) - LN2) / Q;
        //return B*(1. + tanhLUT.get(Q*x)) + (M + softplusLUT.get(-2.*M) - LN2) / Q;
        return (M + softplusLUT.get(-2.*M) - LN2) / Q;
    }*/
    
    static inline float softAbs(float x, float Q, float B) {
        const float LN2 = 0.69314718056; //ln(2)
        float M = std::abs(x * Q);
        //return (M + std::log1p(std::exp(-2.*M)) - LN2) / Q;
        //return B*(1. + tanhLUT.get(Q*x)) + (M + softplusLUT.get(-2.*M) - LN2) / Q;
        return B*(1. + tanhLUT.get(Q*x)) + (M + softplusLUT.get(-2.*M) - LN2) / Q;
    }

    double compressor(double x, double Q, double k, double S, double T, double B, double cutoff, Dsp::Filter* biquad, float** biquad_out) {
        // there's no feedback here, so each stage of this could be batched
        double absR = softAbs(x, Q, B);
        //biquad_out[0][0] = static_cast<float>(absR);
        //biquad->process(1, biquad_out);
        double bqR = absR;//biquad_out[0][0];
        return x * gainReduction(bqR, k, S, T);
    }

    void compute(
            std::array<double, HISTORY_SIZE>& l,
            std::array<double, HISTORY_SIZE>& r,
            double Q,
            double k,
            double S,
            double T,
            double B,
            double cutoff
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
        
        for (int i = 0; i < OVERSAMPLE; i++) {
            slew_prev_L =  math::crossfade(slew_prev_L, output_x[0][i], cutoff / 10000.f);
            output_x[0][i] = slew_prev_L;
            slew_prev_R =  math::crossfade(slew_prev_R, output_y[0][i], cutoff / 10000.f);
            output_y[0][i] = slew_prev_R;
        }
        
        //biquad_L->process(OVERSAMPLE, output_x);
        //biquad_R->process(OVERSAMPLE, output_y);
        
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

    /**
     * process
     *
     * Called by VCV Rack to process audio every sample.
     */
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

        if (c_prev != c) {
            //biquad_input_L.setParameters(dsp::TBiquadFilter<double>::Type::LOWPASS, c, biquad_Q, biquad_gain);
            //biquad_input_R.setParameters(dsp::TBiquadFilter<double>::Type::LOWPASS, c, biquad_Q, biquad_gain);
            biquad_params[1] = c; // cutoff frequency
            biquad_L->setParams(biquad_params);
            biquad_R->setParams(biquad_params);
        }


        compute(history_x, history_y, q, k, s, t, b, c);

        // Step 4: Set the output voltage based on y0
        outputs[OUTPUT_L].setVoltage(out_L);
        outputs[OUTPUT_R].setVoltage(out_R);
    }
};

//////////////////////////
// Module Widget
//////////////////////////

/**
 * FirstCompressorWidget
 *
 * The GUI widget for the FirstCompressorModule.
 * struct WaterTableWidget : ModuleWidget {
	WaterTableWidget(WaterTable* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/WaterTable.svg")));
 */
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

        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 120), module, FirstCompressorModule::PARAM_K));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 160), module, FirstCompressorModule::PARAM_Q));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 200), module, FirstCompressorModule::PARAM_S));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 240), module, FirstCompressorModule::PARAM_T));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 280), module, FirstCompressorModule::PARAM_C));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 320), module, FirstCompressorModule::PARAM_B));
    }
};

// Register the module with VCV Rack
Model *modelFirstCompressor = createModel<FirstCompressorModule, FirstCompressorWidget>("FreeSurface-FirstCompressor");
