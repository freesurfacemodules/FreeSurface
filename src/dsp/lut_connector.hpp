#pragma once
// Cubic-Hermite LUT core for the line connector (the FAST mode).
//
// 512 uniform segments over [xL, xR]; the joins land exactly on the
// boundary knots, which carry the exact endpoint value and slope, so the
// C1 join with the analytic exterior lines is preserved exactly.  Knot
// derivatives come from the exact factored-form derivative.  Measured
// (log 92.5-92.6): approximation at the float32 table floor (-142 dB re
// full scale) at every degree up to 255; alias floors identical to the
// exact evaluator within 0.3 dB; evaluation ~43 ns/sample flat in
// degree.  Smaller tables were measured and rejected: evaluation cost is
// interpolant-bound, not table-size-bound, so shrinking N buys nothing
// but accuracy loss.
//
// The build is amortized: beginBuild() snapshots the connector, step(k)
// computes k knots into the inactive table, and completion swaps it in
// atomically from the audio thread's view (everything runs on the audio
// thread; the previous table stays active meanwhile).
#include <array>
#include "connector.hpp"

namespace polydist {

struct LutCore {
    static constexpr int kSegments = 512;

    struct Table {
        float xL = -1.f, xR = 1.f, inv_dx = (float)(kSegments / 2.0);
        float lv = -1.f, ls = 0.f, rv = 1.f, rs = 0.f;
        std::array<float, 4 * kSegments> coef{};

        float eval(float x) const {
            if (x < xL) return lv + ls * (x - xL);
            if (x > xR) return rv + rs * (x - xR);
            float s = (x - xL) * inv_dx;
            int i = (int)s;
            if (i >= kSegments) i = kSegments - 1;
            const float f = s - (float)i;
            const float* q = &coef[4 * i];
            return ((q[3] * f + q[2]) * f + q[1]) * f + q[0];
        }
        void apply_block(float* out, const float* in, int n) const {
            for (int i = 0; i < n; ++i) out[i] = eval(in[i]);
        }
    };

    bool ready() const { return active_ >= 0; }
    bool building() const { return pos_ >= 0; }
    const Table& table() const { return tables_[active_]; }

    void beginBuild(const LineConnector& c) {
        src_ = c;
        pos_ = 0;
        Table& b = back();
        b.xL = c.fxL; b.xR = c.fxR;
        b.inv_dx = (float)(kSegments / (c.xR - c.xL));
        b.lv = c.flv; b.ls = c.fls; b.rv = c.frv; b.rs = c.frs;
        v0_ = c.left_value;                       // exact at the join
        d0_ = c.interiorDerivative(c.xL);
    }

    // Compute up to `knots` further knots; returns true on completion
    // (the new table becomes active).
    bool step(int knots) {
        if (pos_ < 0) return false;
        Table& b = back();
        const double dx = (src_.xR - src_.xL) / kSegments;
        for (int k = 0; k < knots && pos_ < kSegments; ++k, ++pos_) {
            const bool last = pos_ == kSegments - 1;
            const double x1 = src_.xL + dx * (pos_ + 1);
            const double v1 = last ? src_.right_value : src_(x1);
            const double d1 = src_.interiorDerivative(last ? src_.xR : x1);
            const double m0 = d0_ * dx, m1 = d1 * dx;
            float* q = &b.coef[4 * pos_];
            q[0] = (float)v0_;
            q[1] = (float)m0;
            q[2] = (float)(-3 * v0_ + 3 * v1 - 2 * m0 - m1);
            q[3] = (float)(2 * v0_ - 2 * v1 + m0 + m1);
            v0_ = v1; d0_ = d1;
        }
        if (pos_ >= kSegments) {
            active_ = backIndex();
            pos_ = -1;
            return true;
        }
        return false;
    }

private:
    int backIndex() const { return active_ == 0 ? 1 : 0; }
    Table& back() { return tables_[backIndex()]; }

    Table tables_[2];
    int active_ = -1;
    LineConnector src_;
    int pos_ = -1;
    double v0_ = 0.0, d0_ = 0.0;
};

} // namespace polydist
