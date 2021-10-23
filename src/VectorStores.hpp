#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include "samplerate.h"

#define unique_resampler_stereo_float std::unique_ptr<StereoFloatResamplerBuffer>
#define resampler_stereo_float StereoFloatResamplerBuffer

/* Buffer with two stages:
   A write stage in which the buffer is repeatedly pushed new samples
   in arrays with size CHANNELS. The buffer will resize to fit data on demand,
   and maintain the double buffer, but portions of the buffer will remain
   uninitalized.

   After writing is complete, the finalize() method must be called. The buffer
   will be compacted to fit the stored data.

   At this point, the buffer will no longer resize on demand, and will behave
   like a fixed size double ring buffer. The buffer can still be written to
   using the endData/endIncr pattern.

  */

template <typename T>
struct InterleavedExpandingDoubleRingBuffer {
    size_t S = 0;
    T* data;
    size_t start = 0;
    size_t end = 0;
    size_t channels = 0;
    bool finalized = false;

    InterleavedExpandingDoubleRingBuffer(size_t size, size_t chans) {
        channels = chans;
        S = size * channels;
        data = new T[S * 2];
    }

    ~InterleavedExpandingDoubleRingBuffer() {
        delete [] data;
    }

    void finalize() {
        contract();
        finalized = true;
    }

    bool isFinalized() {
        return finalized;
    }

    size_t mask(size_t i) const {
        if (S == 0) {
            return 0;
        } else {
            return i % S;
        }
    }

    void expand() {
        if (finalized) return;
        if (S == 0) {
            S = channels;
        }
        size_t new_S = S * 2;
        T* new_data = new T[new_S * 2];
        memcpy(new_data, data, sizeof(T) * S);
        memcpy(&new_data[new_S], &data[S], sizeof(T) * S);
        delete [] data;
        data = new_data;
        //start = mask(start);
        start = 0;
        S = new_S;
    }

    void zeroPad() {
        if (end < S) {
            size_t len = S - end;
            std::vector<T> zeros(len);
            memcpy(&data[end], zeros.data(), sizeof(T) * len);
            memcpy(&data[end + S], zeros.data(), sizeof(T) * len);
            end = S;
        }
    }

    // Repeat data in partially-full buffer to pad it to full size
    void padRepeatingTo(size_t N) {
        if (finalized) return;
        if ((N*channels) < end) return;
        size_t new_S = N * channels;
        T* new_data = new T[new_S * 2];
        size_t offset = 0;
        while (offset < new_S) {
            size_t copy_len = std::min(new_S-offset, end);
            memcpy(&new_data[offset], data, sizeof(T) * copy_len);
            offset += end;
        }
        memcpy(&new_data[new_S], new_data, sizeof(T) * new_S);
        delete [] data;
        data = new_data;
        end = new_S;
        S = new_S;
        //start = mask(start);
        start = 0;
    }

    void contract() {
        if (finalized) return;
        if (end == S) {
            return;
        } else if (end == 0) {
            delete[] data;
            data = new T[0];
            S = 0;
            start = 0;
        } else {
            size_t new_S = end;
            T *new_data = new T[new_S * 2];
            memcpy(new_data, data, sizeof(T) * new_S);
            memcpy(&new_data[new_S], &data[S], sizeof(T) * new_S);
            delete[] data;
            data = new_data;
            S = new_S;
            //start = mask(start);
            start = 0;
        }
    }

    size_t getInternalSize() {
        return S;
    }

    size_t getInternalFrames() {
        return S / channels;
    }

    // MUST push an array of samples equal in length to the number of channels
    void push(T* t) {
        if (end >= S && !finalized) {
            expand();
        }
        size_t masked_end = end;
        if (finalized) {
            masked_end = mask(end);
        }
        memcpy(&data[masked_end], t, sizeof(T) * channels);
        memcpy(&data[masked_end + S], t, sizeof(T) * channels);
        end += channels;
    }

    // delete everything and un-finalize, but leave channel number intact
    void initEmpty() {
        finalized = false;
        start = 0;
        end = 0;
        S = 0;
        delete [] data;
        data = new T[0];
    }


    // ONLY call when initializing with enough data to fill the buffer
    // init the buffer from input data, reset start and end
    void initCopy(T* tocopy) {
        if (finalized) return;
        memcpy(data, tocopy, sizeof(T) * S);
        memcpy(&data[S], tocopy, sizeof(T) * S);
        start = 0;
        end = S;
    }

    // *ONLY* call when initializing with less data than can fill the buffer.
    // Will repeat the initial data. N is the size of the copied buffer.
    void initCopyUndersized(T* tocopy, size_t N) {
        if (finalized) return;
        size_t offset = 0;
        while (offset < S) {
            size_t copy_len = std::min(S-offset, N);
            memcpy(&data[offset], tocopy, sizeof(T) * copy_len);
            offset += N;
        }
        memcpy(&data[S], data, sizeof(T) * S);
        start = 0;
        end = S;
    }

    /**
     * BEGIN FUNCTIONS THAT ARE ONLY VALID WHEN BUFFER IS FINALIZED
     */
    T* shift() {
        T* todata = &data[mask(start)];
        start += channels;
        return todata;
    }
    void clear() {
        start = end;
    }
    bool empty() const {
        return start == end;
    }
    bool full() const {
        return end - start == S;
    }

    // returns size in frames
    size_t size() const {
        return (end - start) / channels;
    }

    // returns size in frames
    size_t capacity() const {
        return (S / channels) - size();
    }

    /** Returns a pointer to S consecutive elements for appending.
    If any data is appended, you must call endIncr afterwards.
    Pointer is invalidated when any other method is called.
    */
    T* endData() {
        return &data[mask(end)];
    }

    void endIncr(size_t n) {
        size_t n_len = n * channels;
        size_t e = mask(end);
        size_t e1 = e + n_len;
        size_t e2 = (e1 < S) ? e1 : S;
        // Copy data forward
        memcpy(&data[S + e], &data[e], sizeof(T) * (e2 - e));

        if (e1 > S) {
            // Copy data backward from the doubled block to the main block
            memcpy(data, &data[S], sizeof(T) * (e1 - S));
        }
        end += n_len;
    }

    /** Returns a pointer to S consecutive elements for consumption
    If any data is consumed, call startIncr afterwards.
    */
    const T* startData() const {
        return &data[mask(start)];
    }
    void startIncr(size_t n) {
        start += n * channels;
    }

    /**
     * END FUNCTIONS THAT ARE ONLY VALID WHEN BUFFER IS FINALIZED
     */
};

template <typename T, size_t CHANNELS>
struct ResamplerBuffer {
    InterleavedExpandingDoubleRingBuffer<T>* input;
    InterleavedExpandingDoubleRingBuffer<T>* output;
    ResamplerBuffer(size_t output_buffer_size) {
        input = new InterleavedExpandingDoubleRingBuffer<T>(0, CHANNELS);
        output = new InterleavedExpandingDoubleRingBuffer<T>(output_buffer_size, CHANNELS);
        output->zeroPad();
        output->finalize();
    }
    ~ResamplerBuffer() {
        delete input;
        delete output;
    }
};

struct StereoSample {
	float x;
	float y;
	bool bufferEmpty;
};

struct StereoFloatResamplerBuffer : ResamplerBuffer<float, 2> {
    SRC_STATE* src;
    float resampleRatio = 1.0;
    size_t input_buffer_size_minimum;
	StereoFloatResamplerBuffer(size_t output_buffer_size, size_t input_buffer_size_minimum);
    ~StereoFloatResamplerBuffer() {
        src_delete(src);
    }
	void pushInput(float x, float y);
	StereoSample shiftOutput();
    void finalize();
    void reset();
    bool ready() const;
    void resample();
    void setResampleRatio(float ratio);
    float size() const;
};
