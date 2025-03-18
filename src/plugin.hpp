#include "rack.hpp"
#include "Utility.hpp"
#include "CVParamInput.hpp"
#include "VektronixComponents.hpp"
#include "WaterTableComponents.hpp"
#include "WaterTable2Display.hpp"

using simd::float_4;
using simd::int32_4;

using namespace rack;

extern Plugin* pluginInstance;

extern Model* modelWaterTable;
extern Model* modelWaterTable2;
//extern Model* modelAliasFreeDistortion;
extern Model* modelStereoToMonoFFT;
extern Model* modelFirstCompressor;
extern Model* modelPhiReverb;
extern Model* modelKalmanPitchTracker;

