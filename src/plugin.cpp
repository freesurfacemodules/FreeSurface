#include "plugin.hpp"


Plugin* pluginInstance;

void init(rack::Plugin* p) {
	pluginInstance = p;

	p->addModel(modelWaterTable);
    p->addModel(modelWaterTable2);
    //p->addModel(modelAliasFreeDistortion);
}
