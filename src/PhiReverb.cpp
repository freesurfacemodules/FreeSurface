#include "rack.hpp"
#include "../src/plugin.hpp"

using namespace rack;

// Ensure M_PI is defined.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constants
const int BUFFER_SIZE = 4096;
const float BASE_DELAY = 179.0f;
const float PHI = 1.618f;
const int NUM_TAPS = 4;
const float MAX_BASE_DELAY = 511; // (BUFFER_SIZE / (MAX_INTERVAL^3)) - 1
const float MAX_INTERVAL = 2;

// ---------------------------------------------------------------------------
// Simple POD struct for complex numbers
struct Complex {
    float real;
    float imag;
};

// Basic complex arithmetic (as inline functions)
inline Complex add(const Complex &a, const Complex &b) {
    return { a.real + b.real, a.imag + b.imag };
}

inline Complex multiply(const Complex &a, const Complex &b) {
    return { a.real * b.real - a.imag * b.imag,
             a.real * b.imag + a.imag * b.real };
}

inline Complex scale(const Complex &a, float s) {
    return { a.real * s, a.imag * s };
}

inline Complex fromPolar(float magnitude, float phase) {
    return { magnitude * cosf(phase), magnitude * sinf(phase) };
}

// ---------------------------------------------------------------------------
// A simple ring (circular) buffer for Complex samples.
struct CircularBuffer {
    Complex buffer[BUFFER_SIZE];
    int writeIndex;

    CircularBuffer() : writeIndex(0) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i] = { 0.0f, 0.0f };
        }
    }

    // Write a sample and advance the write pointer.
    void push(const Complex &sample) {
        buffer[writeIndex] = sample;
        writeIndex = (writeIndex + 1) % BUFFER_SIZE;
    }

    // Read a sample from a given delay (in samples)
    Complex readDelay(int delay) {
        int readIndex = writeIndex - delay;
        if (readIndex < 0)
            readIndex += BUFFER_SIZE;
        return buffer[readIndex];
    }
};

// ---------------------------------------------------------------------------
// First-order Thiran Allpass Filter to compensate for fractional delay.
// Difference equation:
//   y(n) = C * (y(n-1) - x(n)) + x(n-1)
// where C = (d - 1) / (d + 1), and d is the fractional delay.
struct ThiranAllpass {
    float C;
    Complex prevInput;
    Complex prevOutput;

    ThiranAllpass(float d)
        : C((d - 1.0f) / (d + 1.0f)),
          prevInput({ 0.0f, 0.0f }),
          prevOutput({ 0.0f, 0.0f }) { }

    // Process one sample (x is complex) and update state.
    Complex process(const Complex &x) {
        Complex y;
        y.real = C * (prevOutput.real - x.real) + prevInput.real;
        y.imag = C * (prevOutput.imag - x.imag) + prevInput.imag;
        prevInput = x;
        prevOutput = y;
        return y;
    }
    
    void setFractionalDelay(float d) {
        C = (d - 1.0f) / (d + 1.0f);
    }
};

// ---------------------------------------------------------------------------
// DelayTap encapsulates one delay line with a circular buffer and a Thiran filter.
struct DelayTap {
    CircularBuffer buffer;
    int delaySamples;        // Integer delay (in samples)
    ThiranAllpass filter;    // Thiran filter for fractional delay

    // Constructor: delaySamples is integer part; fractionalDelay is the fractional part.
    DelayTap(int delaySamples = 179, float fractionalDelay = 0.f)
        : delaySamples(delaySamples),
          filter(fractionalDelay) { }

    // Read the delayed sample and apply the Thiran allpass filter.
    Complex getTap() {
        Complex delayedSample = buffer.readDelay(delaySamples);
        return filter.process(delayedSample);
    }

    // Push a new sample into the delay line.
    void push(const Complex &sample) {
        buffer.push(sample);
    }
    
    void setDelay(int d) {
        delaySamples = d;
    }
    
    void setFractionalDelay(float fractionalDelay) {
        filter.setFractionalDelay(fractionalDelay);
    }
    
    void setDelay(float delay) {
        int intDelay = (int)floorf(delay);
        float fracDelay = delay - intDelay;
        delaySamples = intDelay;
        filter.setFractionalDelay(fracDelay);
    }
};

// ---------------------------------------------------------------------------
// One-pole lowpass filter for complex signals.
// Uses the difference equation:
//   y(n) = (1 - a) * x(n) + a * y(n-1)
// where the coefficient a = exp(-2*pi*cutoff/sampleRate).
struct OnePoleLPF {
    float a; // Coefficient (smoothing factor)
    Complex prevOutput;

    OnePoleLPF(float cutoff = 20000.f, float sampleRate = 48000.f) {
        setCutoff(cutoff, sampleRate);
        prevOutput = { 0.0f, 0.0f };
    }

    void setCutoff(float cutoff, float sampleRate) {
        // Compute the coefficient: a = exp(-2*pi*cutoff/sampleRate)
        a = expf(-2.0f * M_PI * cutoff / sampleRate);
    }

    // Process one complex sample.
    Complex process(const Complex &x) {
        Complex y;
        y.real = (1.0f - a) * x.real + a * prevOutput.real;
        y.imag = (1.0f - a) * x.imag + a * prevOutput.imag;
        prevOutput = y;
        return y;
    }
};

// ---------------------------------------------------------------------------
// Update the 4x4 circulant unitary matrix.
// The first row is defined by:
//   c_j = (1/4) * sum_{k=0}^{3} e^(i theta_k) * e^(2pi i j k/4),   for j = 0,...,3.
// The full circulant matrix U is built by cyclically shifting this row.
static void updateUnitaryMatrix(const float theta[NUM_TAPS], Complex U[NUM_TAPS][NUM_TAPS]) {
    Complex firstRow[NUM_TAPS];
    const int n = NUM_TAPS;

    for (int j = 0; j < n; j++) {
        Complex sum = { 0.0f, 0.0f };
        for (int k = 0; k < n; k++) {
            // e^(i theta_k)
            Complex phase = fromPolar(1.0f, theta[k]);
            // e^(2pi i * j * k / n)
            float angle = 2.0f * M_PI * j * k / n;
            Complex twiddle = fromPolar(1.0f, angle);
            // Multiply and accumulate
            sum = add(sum, multiply(phase, twiddle));
        }
        // Scale by 1/n.
        firstRow[j] = scale(sum, 1.0f / n);
    }

    // Build the circulant matrix: each row is a cyclic shift of firstRow.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = j - i;
            if (index < 0)
                index += n;
            U[i][j] = firstRow[index];
        }
    }
}


// ---------------------------------------------------------------------------
// Feedback Delay Network (FDN) Reverb class.
struct FDNReverb {
    // Four delay taps.
    DelayTap taps[NUM_TAPS];
    // 4x4 unitary matrix.
    Complex U[NUM_TAPS][NUM_TAPS];
    // Four phase parameters (controlled by knobs, range 0 to 2pi).
    float theta[NUM_TAPS];
    
    float fbscale = 0.9;
    
    // One-pole lowpass filters on the feedback path.
    OnePoleLPF lpf[NUM_TAPS];
    // Sample rate (in Hz) used for filter coefficient calculation.
    float sampleRate;
    // Cutoff frequency for the LPF (in Hz).
    float lpfCutoff;
    
    float baseDelay = BASE_DELAY;
    float interval = PHI;

    // Constructor: initialize taps and unitary matrix.
    FDNReverb(float base_delay = BASE_DELAY, float sampleRate_ = 44100.0f, float cutoff = 5000.0f, float interval_ = PHI) : 
            sampleRate(sampleRate_), 
            lpfCutoff(cutoff),
            interval(interval_)
        {
        // Initialize theta to default values (e.g., 0 radians).
        for (int i = 0; i < NUM_TAPS; i++) {
            theta[i] = 0.0f;
        }

        // For each tap, compute delay parameters.
        // Delay = BASE_DELAY * phi^(tap_number)
        // Integer part is floor(delay), fractional part is delay - floor(delay).
        for (int i = 0; i < NUM_TAPS; i++) {
            float delay = base_delay * powf(interval, i);
            taps[i].setDelay(delay);
        }

        // Initialize the unitary matrix.
        updateUnitaryMatrix(theta, U);
        
        
        // Initialize one-pole LPF for each feedback path.
        for (int i = 0; i < NUM_TAPS; i++) {
            lpf[i].setCutoff(lpfCutoff, sampleRate);
        }
    }
    
    void setBaseDelay(float baseDelay_, float interval_) {
        baseDelay = baseDelay_;
        interval = interval_;
        for (int i = 0; i < NUM_TAPS; i++) {
            float delay = baseDelay * powf(interval, i);
            taps[i].setDelay(delay);
        }
    }

    // Set one of the four phase parameters (knob input, in radians).
    void setTheta(int index, float value) {
        if (index >= 0 && index < NUM_TAPS) {
            theta[index] = value;
            updateUnitaryMatrix(theta, U);
        }
    }
    
    // Set all four phase parameters (knob input, in radians).
    void setTheta(float a, float b, float c, float d) {
        theta[0] = a;
        theta[1] = b;
        theta[2] = c;
        theta[3] = d;
        updateUnitaryMatrix(theta, U);
    }
    
    void setInterval(float interval_) {
        setBaseDelay(baseDelay, interval_); // reset all derived delay lengths
    }
    
    // Set the one-pole lowpass filter cutoff frequency (in Hz).
    void setLPFCutoff(float cutoff) {
        lpfCutoff = cutoff;
        for (int i = 0; i < NUM_TAPS; i++) {
            lpf[i].setCutoff(lpfCutoff, sampleRate);
        }
    }
    
    void setFeedback(float f) {
        fbscale = f;
    }
    
    Complex processSample(float input) {
        Complex inputComplex = { input, 0.0f };
        return processSample(inputComplex);
    };

    // Process one sample.
    // 'input' is a real-valued sample.
    // Returns a Complex value whose real and imaginary parts are routed to the module outputs.
    Complex processSample(Complex input) {
        // 1. Get output from each delay tap.
        Complex tapOutputs[NUM_TAPS];
        for (int i = 0; i < NUM_TAPS; i++) {
            tapOutputs[i] = taps[i].getTap();
        }

        // 2. Multiply the 4-vector of tap outputs by the unitary matrix U.
        Complex feedback[NUM_TAPS];
        for (int i = 0; i < NUM_TAPS; i++) {
            feedback[i] = { 0.0f, 0.0f };
            for (int j = 0; j < NUM_TAPS; j++) {
                feedback[i] =  add(feedback[i], scale(multiply(U[i][j], tapOutputs[j]), fbscale));
            }
        }
        
        // 3. Process each feedback channel through its one-pole lowpass filter.
        for (int i = 0; i < NUM_TAPS; i++) {
            feedback[i] = lpf[i].process(feedback[i]);
        }

        // 4. Feed back into the delay lines.
        // For each delay tap, push the new sample, which is the input sample (converted to complex)
        // plus the corresponding feedback component.
        for (int i = 0; i < NUM_TAPS; i++) {
            Complex newSample = add(input, feedback[i]);
            taps[i].push(newSample);
        }

        // 5. Compute the output by taking the dot product of the feedback vector with (1,1,1,1)
        // (i.e. summing all its elements).
        Complex output = { 0.0f, 0.0f };
        for (int i = 0; i < NUM_TAPS; i++) {
            output = add(output, feedback[i]);
        }
        return output;
    }
};

struct PhiReverbModule : Module {
    // Enum identifiers for Params, Inputs, Outputs, Lights
    enum ParamIds {
        PARAM_R0,
        PARAM_R1,
        PARAM_R2,
        PARAM_R3,
        PARAM_T,
        PARAM_F,
        PARAM_D0,
        PARAM_D1,
        PARAM_D2,
        PARAM_I0,
        PARAM_I1,
        PARAM_I2,
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

    float out_L;
    float out_R;
    
    FDNReverb reverb0{179.f};
    FDNReverb reverb1{239.f};
    FDNReverb reverb2{419.f};
    
    // Constructor: Initialize module and history buffer
    PhiReverbModule() {
        config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
        configParam(PARAM_R0, -M_PI, M_PI, 0.5f, "K");
        configParam(PARAM_R1, -M_PI, M_PI, 10.f, "Q");
        configParam(PARAM_R2, -M_PI, M_PI, 2.f, "S");
        configParam(PARAM_R3, -M_PI, M_PI, -12.f, "T");
        configParam(PARAM_T, 0.0f, 0.99999f, 0.9f, "C");
        configParam(PARAM_F, 1.0f, 20000.0f, 10000.0f, "R");
        configParam(PARAM_D0, 0.f, MAX_BASE_DELAY, 179.f, "D0");
        configParam(PARAM_D1, 0.f, MAX_BASE_DELAY, 239.f, "D1");
        configParam(PARAM_D2, 0.f, MAX_BASE_DELAY, 419.f, "D2");
        configParam(PARAM_I0, 0.f, MAX_INTERVAL, PHI, "I0");
        configParam(PARAM_I1, 0.f, MAX_INTERVAL, PHI, "I1");
        configParam(PARAM_I2, 0.f, MAX_INTERVAL, PHI, "I2");
    }
    
    float r0_prev = 0.f;
    float r1_prev = 0.f;
    float r2_prev = 0.f;
    float r3_prev = 0.f;
    
    float d0_prev = 0.f;
    float d1_prev = 0.f;
    float d2_prev = 0.f;
    
    float i0_prev = 0.f;
    float i1_prev = 0.f;
    float i2_prev = 0.f;
    
    float f_prev = 0.f;
    float t_prev = 0.f;

    void process(const ProcessArgs &args) override {
        // Step 1: Read the current input voltage
        float input_x = inputs[INPUT_X].getVoltage();

        float r0 = params[PARAM_R0].getValue();
        float r1 = params[PARAM_R1].getValue();
        float r2 = params[PARAM_R2].getValue();
        float r3 = params[PARAM_R3].getValue();
        
        float t = params[PARAM_T].getValue();
        float f = params[PARAM_F].getValue();
        
        float d0 = params[PARAM_D0].getValue();
        float d1 = params[PARAM_D1].getValue();
        float d2 = params[PARAM_D2].getValue();
        float i0 = params[PARAM_I0].getValue();
        float i1 = params[PARAM_I1].getValue();
        float i2 = params[PARAM_I2].getValue();
        
        if (r0_prev != r0 || r1_prev != r1 || r2_prev != r2 || r3_prev != r3) {
            reverb0.setTheta(r0, r1, r2, r3);
            reverb1.setTheta(r0, r1, r2, r3);
            reverb2.setTheta(r0, r1, r2, r3);
        }
        
        if (d0_prev != d0 || i0_prev != i0) {
            reverb0.setBaseDelay(d0, i0);
        }
        
        if (d1_prev != d1 || i1_prev != i1) {
            reverb1.setBaseDelay(d1, i1);
        }
        
        if (d2_prev != d2 || i2_prev != i2) {
            reverb2.setBaseDelay(d2, i2);
        }
        
        if (t_prev != t) {
            reverb0.setFeedback(t);     
            reverb1.setFeedback(t);    
            reverb2.setFeedback(t);       
        }
        
        if (f_prev != f) {
            reverb0.setLPFCutoff(f);
            reverb1.setLPFCutoff(f);  
            reverb2.setLPFCutoff(f);            
        }

        float sf = (1.0f - t)/0.08; // need to figure out a better empirical scaling relationship
        
        Complex result0 = reverb0.processSample(input_x);
        Complex result1 = reverb1.processSample(input_x);
        Complex result2 = reverb2.processSample(scale(add(result0,result1), sf));
        
        // Step 4: Set the output voltage based on y0
        outputs[OUTPUT_L].setVoltage(sf * result2.real);
        outputs[OUTPUT_R].setVoltage(sf * result2.imag);
        
        r0_prev = r0;
        r1_prev = r1;
        r2_prev = r2;
        r3_prev = r3;
    
        d0_prev = d0;
        d1_prev = d1;
        d2_prev = d2;
    
        i0_prev = i0;
        i1_prev = i1;
        i2_prev = i2;
    
        t_prev = t;
        f_prev = f;

    }
};

//////////////////////////
// Module Widget
//////////////////////////
struct PhiReverbWidget : ModuleWidget {
    PhiReverbWidget(PhiReverbModule *module) {
        setModule(module);

        // Set the panel SVG (Replace with your module's SVG file path)
        setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Norms.svg")));

        // Input Port
        addInput(createInputCentered<PJ301MPort>(Vec(10, 30), module, PhiReverbModule::INPUT_X));
        addInput(createInputCentered<PJ301MPort>(Vec(10, 50), module, PhiReverbModule::INPUT_Y));

        // Output Port
        addOutput(createOutputCentered<PJ301MPort>(Vec(10, 70), module, PhiReverbModule::OUTPUT_L));
        addOutput(createOutputCentered<PJ301MPort>(Vec(10, 90), module, PhiReverbModule::OUTPUT_R));

        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 120), module, PhiReverbModule::PARAM_R0));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 140), module, PhiReverbModule::PARAM_R1));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 160), module, PhiReverbModule::PARAM_R2));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 180), module, PhiReverbModule::PARAM_R3));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 200), module, PhiReverbModule::PARAM_T));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 220), module, PhiReverbModule::PARAM_F));
        
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 240), module, PhiReverbModule::PARAM_D0));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 260), module, PhiReverbModule::PARAM_D1));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 280), module, PhiReverbModule::PARAM_D2));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 300), module, PhiReverbModule::PARAM_I0));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 320), module, PhiReverbModule::PARAM_I1));
        addParam(createParamCentered<VektronixSmallKnobDark>(Vec(10, 340), module, PhiReverbModule::PARAM_I2));
    }
};

// Register the module with VCV Rack
Model *modelPhiReverb = createModel<PhiReverbModule, PhiReverbWidget>("FreeSurface-PhiReverb");
