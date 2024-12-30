#include "rack.hpp"
#include "../src/plugin.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include "pffft.h"

using namespace rack;

struct StereoToMonoFFTModule : Module {
    enum ParamIds {
        NUM_PARAMS
    };
    enum InputIds {
        INPUT_LEFT,
        INPUT_RIGHT,
        NUM_INPUTS
    };
    enum OutputIds {
        OUTPUT_MONO,
        NUM_OUTPUTS
    };
    enum LightIds {
        NUM_LIGHTS
    };

    static const int BLOCK_SIZE = 1024; // FFT size
    static const int OVERLAP_FACTOR = 4;
    static const int HOP_SIZE = BLOCK_SIZE / OVERLAP_FACTOR;

    // Hamming window
    std::vector<float> window;

    // Keep track of how many blocks we've processed
    int blockCount = 0;
    long step = 0;

    // Ring buffers for OLA
    dsp::RingBuffer<float, 65536> ringBuffers[OVERLAP_FACTOR];

    dsp::RingBuffer<float, 65536> ringBufferL[OVERLAP_FACTOR];
    dsp::RingBuffer<float, 65536> ringBufferR[OVERLAP_FACTOR];

    // PFFFT setup and buffers
    PFFFT_Setup* pffftSetup = nullptr;
    float* inLeft = nullptr;
    float* inRight = nullptr;
    float* outLeft = nullptr;
    float* outRight = nullptr;
    float* combined = nullptr;
    float* tmp = nullptr;

    // Per-bin phase accumulator
    std::vector<float> phaseAccum;

    std::vector<float> prevPhaseL;
    std::vector<float> prevPhaseR;

    // Time constant for accumulator updates
    // Adjust this value to control how quickly phase preference changes
    float timeConstant = 0.1f;

    StereoToMonoFFTModule() {
        config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);

        // Hamming window
        window.resize(BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            window[i] = 0.5f - 0.5f * std::cos((2.0f * M_PI * i) / (BLOCK_SIZE - 1)); //0.54f - 0.46f * std::cos((2.0f * M_PI * i) / (BLOCK_SIZE - 1));
        }

        pffftSetup = pffft_new_setup(BLOCK_SIZE, PFFFT_COMPLEX);

        inLeft   = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));
        inRight  = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));
        outLeft  = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));
        outRight = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));
        combined = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));
        tmp      = (float*)pffft_aligned_malloc(2 * BLOCK_SIZE * sizeof(float));

        std::memset(inLeft, 0, 2*BLOCK_SIZE*sizeof(float));
        std::memset(inRight, 0, 2*BLOCK_SIZE*sizeof(float));
        std::memset(outLeft, 0, 2*BLOCK_SIZE*sizeof(float));
        std::memset(outRight,0, 2*BLOCK_SIZE*sizeof(float));
        std::memset(combined,0, 2*BLOCK_SIZE*sizeof(float));
        std::memset(tmp,    0, 2*BLOCK_SIZE*sizeof(float));

        // Initialize phase accumulator
        phaseAccum.resize(BLOCK_SIZE, 0.0f);
        prevPhaseL.resize(BLOCK_SIZE, 0.0f);
        prevPhaseR.resize(BLOCK_SIZE, 0.0f);
    }

    ~StereoToMonoFFTModule() {
        if (pffftSetup) pffft_destroy_setup(pffftSetup);
        pffft_aligned_free(inLeft);
        pffft_aligned_free(inRight);
        pffft_aligned_free(outLeft);
        pffft_aligned_free(outRight);
        pffft_aligned_free(combined);
        pffft_aligned_free(tmp);
    }

    inline bool inRange(float x, float a, float b) {
        return (a <= b) ? (x >= a && x <= b) : (x >= b && x <= a);
    }

    float closestPhase(float phaseOld, float phaseNew) {

        float diff = phaseNew - phaseOld;
        // ex: phaseNew = 0.1, phaseOld = 2*pi-0.1, diff = -2*pi+0.2, actual: 0.2
        float ap0 = std::abs(diff + 2.f * M_PI);
        // ex: phaseNew = 2*pi-0.1, phaseOld = 0.1, diff = 2*pi-0.2, actual: -0.2
        float ap1 = std::abs(diff - 2.f * M_PI);
        float ap2 = std::abs(diff);
        if (ap0 < ap2) {
            return ap0;
        } else if (ap1 < ap2) {
            return -ap1;
        } else {
            return diff;
        }
    }

    void process_block() {
        int currentBufferIndex = blockCount % OVERLAP_FACTOR;

        // Window input and load into inLeft/inRight
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float wl = ringBufferL[currentBufferIndex].shift() * window[i];
            float wr = ringBufferR[currentBufferIndex].shift() * window[i];
            inLeft[2*i]   = wl;
            inLeft[2*i+1] = 0.0f;
            inRight[2*i]   = wr;
            inRight[2*i+1] = 0.0f;
        }

        // Forward FFT
        pffft_transform(pffftSetup, inLeft, outLeft, tmp, PFFFT_FORWARD);
        pffft_transform(pffftSetup, inRight, outRight, tmp, PFFFT_FORWARD);



        // Combine frequency bins with per-bin phase preference
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float Lr = outLeft[2*k];
            float Li = outLeft[2*k+1];
            float Rr = outRight[2*k];
            float Ri = outRight[2*k+1];

            float magL = std::sqrt(Lr*Lr + Li*Li);
            float magR = std::sqrt(Rr*Rr + Ri*Ri);

            float phaseL = std::atan2(Li, Lr);
            float phaseR = std::atan2(Ri, Rr);

            float pdiffL = closestPhase(phaseL, prevPhaseL[k]);
            float pdiffR = closestPhase(phaseR, prevPhaseR[k]);

            // Update phase accumulator
            // If magL > magR, phaseAccum moves positive, favoring left
            // If magR > magL, phaseAccum moves negative, favoring right
            //phaseAccum[k] += timeConstant * (magL - magR);

            // Pick phase based on accumulator sign
            //float chosenPhase = (phaseAccum[k] > 0.0f) ? phaseL : phaseR;
            //float chosenPhase = phaseL;

            // Combine magnitudes as before
            float mag = 0.5f * (magL + magR);
            float Cr = (Lr + Rr)*0.5f;
            float Ci = (Li + Ri)*0.5f;
            float nrm = 1.f / (std::sqrt(Cr*Cr + Ci*Ci) + 1e-6f);
            Cr = nrm * mag * Cr;
            Ci = nrm * mag * Ci;
            //float mag = magL;

            float phaseC = std::atan2(Ci, Cr);

            float pdiffC = closestPhase(phaseC, phaseAccum[k]);
            //if (!inRange(pdiffC, pdiffL, pdiffR)) {
                /*Ci *= -1.f;
                Cr *= -1.f;
                phaseC = std::atan2(Ci, Cr);*/
                //phaseC += (pdiffL + pdiffR) * 0.5f;
                phaseC = phaseAccum[k] + (pdiffL + pdiffR) * 0.5f;
                //Cr = mag * std::cos(phaseC);
                //Ci = mag * std::sin(phaseC);
            //}
            phaseAccum[k] = phaseC;
            prevPhaseL[k] = phaseL;
            prevPhaseR[k] = phaseR;

            float outReal = Cr;
            float outImag = Ci;


            //float outReal = mag * std::cos(chosenPhase);
            //float outImag = mag * std::sin(chosenPhase);

            combined[2*k] = outReal;
            combined[2*k+1] = outImag;
        }

        // Inverse FFT
        pffft_transform(pffftSetup, combined, combined, tmp, PFFFT_BACKWARD);

        // Normalize and apply Hamming window again (for OLA)
        float norm = 1.0f / (float)BLOCK_SIZE;
        std::vector<float> processedBlock(BLOCK_SIZE, 0.0f);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float val = combined[2*i] * norm * window[i];
            processedBlock[i] = val;
        }

        // OLA with 4 ring buffers.
        ringBuffers[currentBufferIndex].pushBuffer(processedBlock.data(), BLOCK_SIZE);

        blockCount++;
    }

    void process(const ProcessArgs &args) override {
        float inL = inputs[INPUT_LEFT].isConnected() ? inputs[INPUT_LEFT].getVoltage() : 0.0f;
        float inR = inputs[INPUT_RIGHT].isConnected() ? inputs[INPUT_RIGHT].getVoltage() : 0.0f;

        for (int i = 0; i < OVERLAP_FACTOR; i++) {
            if (step >= i * HOP_SIZE) {
                ringBufferL[i].push(inL);
                ringBufferR[i].push(inR);
            }
        }

        step++;

        if ((step >= BLOCK_SIZE) && (step % HOP_SIZE == 0)) {
            process_block();
        }

        // Sum from all 4 ring buffers
        float outSample = 0.0f;
        for (int i = 0; i < OVERLAP_FACTOR; i++) {
            if (!ringBuffers[i].empty()) {
                outSample += ringBuffers[i].shift();
            }
        }

        outputs[OUTPUT_MONO].setVoltage(outSample);
    }
};

struct StereoToMonoFFTWidget : ModuleWidget {
    StereoToMonoFFTWidget(StereoToMonoFFTModule* module) {
        setModule(module);
        setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Causality.svg")));

        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(10, 20)), module, StereoToMonoFFTModule::INPUT_LEFT));
        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(30, 20)), module, StereoToMonoFFTModule::INPUT_RIGHT));

        addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(20, 60)), module, StereoToMonoFFTModule::OUTPUT_MONO));
    }
};

Model* modelStereoToMonoFFT = createModel<StereoToMonoFFTModule, StereoToMonoFFTWidget>("FreeSurface-StereoToMonoFFT");