#pragma once


#include <memory>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <stdexcept>
#include "VectorStores.hpp"
#include "math.hpp"

//#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"

class SvgToVector
{
    private:
        NSVGimage* g_image;
	    float view[4], size_x, size_y, tol, tolSqr;
        void vertex(unique_resampler_stereo_float &resampler, float x, float y);
        float distPtSeg(float x, float y, float px, float py, float qx, float qy);
        float distSqr(float px, float py, float qx, float qy);
        void cubicBez(unique_resampler_stereo_float &resampler, 
                        float x1, float y1, float x2, float y2,
                        float x3, float y3, float x4, float y4,
                        int level);
        void drawPath(unique_resampler_stereo_float &resampler, float* pts, int npts, char closed);
        void render(unique_resampler_stereo_float &resampler, float width, float height);
        void destroy();

    public:
        SvgToVector(float size_x, float size_y, float tol) {
            this->size_x = size_x;
            this->size_y = size_y;
            this->tol = tol;
            this->tolSqr = tol*tol;
        }
        ~SvgToVector() {
            destroy();
        }

        // throws std::runtime_error if SVG cannot be loaded
	    void loadSvg(const char* path, unique_resampler_stereo_float &resampler);

};