#include "plf_nanotimer.h"
#include <string>
#include "rack.hpp"

enum Marker {
    MARKER_0,
    MARKER_1,
    MARKER_2,
    NUM_MARKERS
};

struct ProfilerDaemon {
    const unsigned long log_interval = 1ul << 24;
    double elapsed[NUM_MARKERS] = {0.0};
    unsigned long calls[NUM_MARKERS] = {0};
    unsigned long total_calls = 0;

    void profile(Marker marker, double time) {
        elapsed[marker] += time;
        calls[marker]++;
        total_calls++;
        if ((total_calls & (log_interval - 1)) == 0) {
            log();
        }
    }

    void log() {
        for (int i = 0; i < NUM_MARKERS; i++) {
            double avg = calls[i] > 0 ? elapsed[i] / calls[i] : 0.0;
            DEBUG("Marker %d called %d times, average call time %f ns", i, calls[i], avg);
        }
    }
} profilerDaemon;

struct Profile
{
    plf::nanotimer timer;
    Marker marker;

    Profile(Marker _marker) : marker(_marker)
    {
        timer.start();
    };

    ~Profile()
    {
        profilerDaemon.profile(marker, timer.get_elapsed_ns());
    }

};