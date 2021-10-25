#include "rack.hpp"
#include "Utility.hpp"
#include "CVParamInput.hpp"
#include "VektronixComponents.hpp"
#include "WaterTableComponents.hpp"
#include "CausalityComponents.hpp"

using simd::float_4;
using simd::int32_4;

using namespace rack;

extern Plugin* pluginInstance;

//extern Model* modelCausality;
//extern Model* modelNorms;
//extern Model* modelMeans;
//extern Model* modelLeakyIntegrator;
extern Model* modelVektronix;
extern Model* modelWaterTable;


