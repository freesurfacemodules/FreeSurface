// This is a modified 'example1.c' from the Nano SVG source distribution.
// The original copyright notice is reproduced below:


//
// Copyright (c) 2013 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#include "SvgToVector.hpp"

using namespace rack;

void SvgToVector::vertex(unique_resampler_stereo_float &resampler, float x, float y) {
	float maxWidth = this->size_x / 2.0;
	float maxHeight = this->size_y / 2.0;
	x = math::rescale(x, this->view[0], this->view[2], -maxWidth, maxWidth);
	// Y is flipped
	y = -math::rescale(y, this->view[1], this->view[3], -maxHeight, maxHeight);
	resampler->pushInput(x, y);
}

float SvgToVector::distSqr(float px, float py, float qx, float qy)
{
	float pqx, pqy;
	pqx = qx-px;
	pqy = qy-py;
	return pqx*pqx + pqy*pqy;
}

// distance from point to point segment
float SvgToVector::distPtSeg(float x, float y, float px, float py, float qx, float qy)
{
	float pqx, pqy, dx, dy, d, t;
	pqx = qx-px;
	pqy = qy-py;
	dx = x-px;
	dy = y-py;
	d = pqx*pqx + pqy*pqy;
	t = pqx*dx + pqy*dy;
	if (d > 0) t /= d;
	if (t < 0) t = 0;
	else if (t > 1) t = 1;
	dx = px + t*pqx - x;
	dy = py + t*pqy - y;
	return dx*dx + dy*dy;
}

// This is modified from the original so that it also subdivides linear segments.
// Subdividing linear segments is useful in an audio context when we are concerned about
// proper sampling.
void SvgToVector::cubicBez(unique_resampler_stereo_float &resampler, 
				float x1, float y1, float x2, float y2,
				float x3, float y3, float x4, float y4,
				int level)
{
	float x12,y12,x23,y23,x34,y34,x123,y123,x234,y234,x1234,y1234;
	float dpt, ds;
	
	if (level > 12) return;

	x12 = (x1+x2)*0.5f;
	y12 = (y1+y2)*0.5f;
	x23 = (x2+x3)*0.5f;
	y23 = (y2+y3)*0.5f;
	x34 = (x3+x4)*0.5f;
	y34 = (y3+y4)*0.5f;
	x123 = (x12+x23)*0.5f;
	y123 = (y12+y23)*0.5f;
	x234 = (x23+x34)*0.5f;
	y234 = (y23+y34)*0.5f;
	x1234 = (x123+x234)*0.5f;
	y1234 = (y123+y234)*0.5f;

	dpt = distPtSeg(x1234, y1234, x1,y1, x4,y4);
	ds = distSqr(x1,y1, x4,y4);
	if (dpt > this->tolSqr || ds > this->tolSqr) {
	//if (dpt > this->tolSqr) {
		cubicBez(resampler, x1,y1, x12,y12, x123,y123, x1234,y1234, level+1); 
		cubicBez(resampler, x1234,y1234, x234,y234, x34,y34, x4,y4, level+1); 
	} else {
		vertex(resampler, x4, y4);
	}
}

void SvgToVector::drawPath(unique_resampler_stereo_float &resampler, float* pts, int npts, char closed)
{
	int i;
	vertex(resampler, pts[0], pts[1]);
	for (i = 0; i < npts-1; i += 3) {
		float* p = &pts[i*2];
		cubicBez(resampler, p[0],p[1], p[2],p[3], p[4],p[5], p[6],p[7], 0);
	}
	if (closed) {
		vertex(resampler, pts[0], pts[1]);
	}
}

void SvgToVector::render(unique_resampler_stereo_float &resampler, float width, float height)
{
	float cx, cy, hw, hh, aspect;
	NSVGshape* shape;
	NSVGpath* path;

	// Fit view to bounds
	cx = this->g_image->width*0.5f;
	cy = this->g_image->height*0.5f;
	hw = this->g_image->width*0.5f;
	hh = this->g_image->height*0.5f;

	if (width/hw < height/hh) {
		aspect = (float)height / (float)width;
		this->view[0] = cx - hw * 1.2f;
		this->view[2] = cx + hw * 1.2f;
		this->view[1] = cy - hw * 1.2f * aspect;
		this->view[3] = cy + hw * 1.2f * aspect;
	} else {
		aspect = (float)width / (float)height;
		this->view[0] = cx - hh * 1.2f * aspect;
		this->view[2] = cx + hh * 1.2f * aspect;
		this->view[1] = cy - hh * 1.2f;
		this->view[3] = cy + hh * 1.2f;
	}
	// Size of one pixel.
	float px = (view[2] - view[1]) / (float)width;
	this->tolSqr = this->tol*px*px;

	for (shape = this->g_image->shapes; shape != NULL; shape = shape->next) {
		for (path = shape->paths; path != NULL; path = path->next) {
			drawPath(resampler, path->pts, path->npts, path->closed);
		}
	}
}

void SvgToVector::loadSvg(const char* path, unique_resampler_stereo_float &resampler) {
	this->g_image = nsvgParseFromFile(path, "px", 96.0f);
	if (this->g_image == NULL) {
		destroy();
		throw std::runtime_error("Error loading SVG");
	}
	render(resampler, this->size_x, this->size_y);
	destroy();
}

void SvgToVector::destroy() {
	nsvgDelete(this->g_image);
}