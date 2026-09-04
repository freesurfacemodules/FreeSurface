#include "plugin.hpp"


Plugin* pluginInstance;

void init(rack::Plugin* p) {
	pluginInstance = p;

	p->addModel(modelWaterTable);
    p->addModel(modelWaterTable2);
    //p->addModel(modelAliasFreeDistortion);
    p->addModel(modelStereoToMonoFFT);
    p->addModel(modelFirstCompressor);
    p->addModel(modelPhiReverb);
    p->addModel(modelKalmanPitchTracker);
    p->addModel(modelKronVerb);
}
