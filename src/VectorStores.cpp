#include "VectorStores.hpp"

#define STEREO_CHANNELS 2

StereoFloatResamplerBuffer::StereoFloatResamplerBuffer(size_t output_buffer_size, size_t input_buffer_size_minimum) : ResamplerBuffer<float, STEREO_CHANNELS>(output_buffer_size) {
	this->input_buffer_size_minimum = input_buffer_size_minimum;
	src = src_new(SRC_SINC_FASTEST, STEREO_CHANNELS, NULL);
	assert(src);
}

void StereoFloatResamplerBuffer::pushInput(float x, float y) {
	std::vector<float> sample {x, y};
	input->push(sample.data());
};

float StereoFloatResamplerBuffer::size() const {
	return input->size();
};

void StereoFloatResamplerBuffer::setResampleRatio(float ratio) {
	this->resampleRatio = ratio;
};

bool StereoFloatResamplerBuffer::ready() const {
	return input->isFinalized();
};

void StereoFloatResamplerBuffer::reset() {
	// the input buffer is wiped and reset to initial state for writing
	input->initEmpty();
	// the output buffer remains the same size, just clear it
	output->clear();
};

void StereoFloatResamplerBuffer::finalize() {
	if (input->size() == 0) { // probably an error, but try not to die here
		pushInput(0.0,0.0);
		input->padRepeatingTo(this->input_buffer_size_minimum);
	} else if (input->size() < output->size()) { // for improbably tiny inputs, repeat the contents
		size_t padded_buffer_len = input->size() * ((this->input_buffer_size_minimum / input->size()) + 1);
		input->padRepeatingTo(padded_buffer_len);
	}
	input->finalize();
};

StereoSample StereoFloatResamplerBuffer::shiftOutput() {
	StereoSample s;
	resample();
	if (!output->empty()) {
		float* out = output->shift();
		s.x = out[0];
		s.y = out[1];
		s.bufferEmpty = false;
	} else {
		s.x = 0.0;
		s.y = 0.0;
		s.bufferEmpty = true;
	}
	return s;
}

void StereoFloatResamplerBuffer::resample() {
	if (output->empty()) {
		SRC_DATA srcData;
		srcData.data_in = (const float*) input->startData();
		srcData.data_out = (float*) output->endData();
		//srcData.input_frames = std::min((long) input->getInternalFrames(), (long) output->getInternalFrames());
		srcData.input_frames = (long) input->getInternalFrames();
		srcData.output_frames = (long) output->capacity();
		srcData.end_of_input = false;
		srcData.src_ratio = this->resampleRatio;
		src_process(src, &srcData);
		input->startIncr(srcData.input_frames_used);
		output->endIncr(srcData.output_frames_gen);
	}
}