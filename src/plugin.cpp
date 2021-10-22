#include "plugin.hpp"


Plugin* pluginInstance;

void init(rack::Plugin* p) {
	pluginInstance = p;

	//p->addModel(modelCausality);
	//p->addModel(modelNorms);
	//p->addModel(modelMeans);
	//p->addModel(modelLeakyIntegrator);
	p->addModel(modelVektronix);
	p->addModel(modelWaterTable);
}
