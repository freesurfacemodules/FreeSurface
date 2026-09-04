// Stand-alone checks of the module's DSP (no Rack dependency):
//  1. connector design vs the Python reference (design/line_connector.py):
//     known closed forms and the endpoint derivative conditions
//  2. Shaper4x with the default connector == the study's verified
//     Local4xPipeline with the clamped d7 core (same stages, same numbers)
//  3. per-sample vs block processing
//  4. alias floor of the 4x scaffold by degree cap and drive (baseline)
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
#include "dsp/connector.hpp"
#include "dsp/lut_connector.hpp"
#include "dsp/shaper4x.hpp"
#include "dsp/local_pipeline.hpp"
#include "generated/lg57_config.hpp"

using Cfg = aadsp::generated::lg57_config;
static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

static bool coeffs_match(const polydist::Polynomial& p, std::vector<double> expect, double tol = 1e-9) {
    for (size_t k = 0; k < expect.size(); ++k) {
        double c = (int)k < p.count ? p.c[k] : 0.0;
        if (std::fabs(c - expect[k]) > tol) return false;
    }
    for (int k = (int)expect.size(); k < p.count; ++k) if (std::fabs(p.c[k]) > tol) return false;
    return true;
}

// Low-degree reference: the general Hermite solver with line conditions
// (the pre-swap production path, trustworthy for caps <= 15).
static polydist::Polynomial reference_interior(double lv, double ls, double rv, double rs, int cap) {
    const int r = (cap - 1) / 2;
    double left[polydist::kMaxConditions], right[polydist::kMaxConditions];
    for (int k = 0; k <= r; ++k) {
        left[k] = k == 0 ? lv : (k == 1 ? ls : 0.0);
        right[k] = k == 0 ? rv : (k == 1 ? rs : 0.0);
    }
    polydist::Polynomial p;
    polydist::hermite_connector(left, r + 1, right, r + 1, -1.0, 1.0, p);
    return p;
}

static double max_diff_vs_reference(double lv, double ls, double rv, double rs, int cap) {
    polydist::LineConnector c;
    polydist::connector_between_lines(lv, ls, rv, rs, cap, c);
    polydist::Polynomial ref = reference_interior(lv, ls, rv, rs, cap);
    double worst = 0;
    for (int i = 0; i <= 400; ++i) {
        double x = -1.0 + 2.0 * i / 400;
        worst = std::max(worst, std::fabs(c(x) - ref(x)));
    }
    return worst;
}

static void test_connector() {
    std::printf("connector design (factored four-basis evaluator)\n");
    // classic closed forms via the reference solver (sanity of the reference)
    check(coeffs_match(reference_interior(-1, 0, 1, 0, 3), {0, 1.5, 0, -0.5}), "reference cap 3: cubic smoothstep (3x - x^3)/2");
    check(coeffs_match(reference_interior(-1, 0, 1, 0, 7), {0, 2.1875, 0, -2.1875, 0, 1.3125, 0, -0.3125}), "reference cap 7: the study's d7 odd smoothstep");
    check(coeffs_match(reference_interior(0, 2, 0, -2, 3), {1, 0, -1, 0}), "reference cap 3 between 2x+2 and -2x+2: 1 - x^2");
    // production evaluator == reference solver where the solver is trustworthy
    double worst = 0;
    for (int cap = 1; cap <= 15; ++cap)
        worst = std::max(worst, max_diff_vs_reference(-0.9, 0.4, 1.2, -0.15, cap));
    std::printf("  production vs reference solver, caps 1-15: max |diff| %.2g\n", worst);
    check(worst < 1e-7, "factored evaluator matches the Hermite solve (caps 1-15, arbitrary lines)");
    check(max_diff_vs_reference(-2, 5, 2, -5, 15) < 1e-7, "  ... and on the hard case (values -+2, slopes +-5)");
    // endpoint conditions and even-cap degeneracy across the full range
    bool all = true;
    polydist::LineConnector c, ce;
    for (int cap = 1; cap <= polydist::kMaxDegreeCap; ++cap) {
        polydist::connector_between_lines(-0.9, 0.4, 1.2, -0.15, cap, c);
        all = all && polydist::verify_line_connector(c)
            && c.interior_degree() == 2 * ((cap - 1) / 2) + 1
            && std::fabs(c(-1.0) - (-0.9)) < 1e-12 && std::fabs(c(1.0) - 1.2) < 1e-12;
    }
    check(all, "caps 1..63: value/slope conditions, degree 2r+1, joins exact");
    // float path vs double path at the top of the float range, hard case
    for (int cap : {63, 127}) {
        polydist::connector_between_lines(-2, 5, 2, -5, cap, c);
        double fworst = 0;
        for (int i = 0; i <= 4000; ++i) {
            float x = -1.f + 2.f * i / 4000;
            fworst = std::max(fworst, std::fabs((double)c.eval(x) - c((double)x)));
        }
        std::printf("  float vs double evaluation at cap %d (deg %d): max |diff| %.2g\n", cap, c.interior_degree(), fworst);
        check(fworst < (cap > 63 ? 1e-4 : 3e-5), "float evaluation stable");
    }
    // above the float limit the audio path is the double path by dispatch
    polydist::connector_between_lines(-2, 5, 2, -5, 256, c);
    check(std::fabs((double)c.eval(0.37f) - c(0.37)) < 1e-6 && c.interior_degree() == 255,
          "cap 256: degree-255 audio path dispatches to double");
    // C0 line
    polydist::connector_between_lines(-1, 0, 1, 0, 1, ce);
    check(std::fabs(ce(0.37) - 0.37) < 1e-12 && ce.smoothness_order == 0, "cap 1: C0 hard clipper y = x");
}

static double bin_db(const std::vector<float>& y, int n0, int n, int k) {
    std::complex<double> acc = 0;
    for (int i = 0; i < n; ++i) {
        double w = 0.5 - 0.5 * std::cos(2 * M_PI * i / n);
        acc += w * (double)y[n0 + i] * std::polar(1.0, -2 * M_PI * k * i / n);
    }
    return 20 * std::log10(std::abs(acc) / (0.5 * n) + 1e-30);
}

static const int FS = 48000, N = 1 << 16, WARM = 4096, TOTAL = WARM + N, K0 = 5987;

static std::vector<float> sine(float amp) {
    const double f0 = (double)K0 * FS / N;
    std::vector<float> x(TOTAL);
    for (int i = 0; i < TOTAL; ++i) x[i] = amp * std::sin(2 * M_PI * f0 * i / FS);
    return x;
}

static double worst_alias(const std::vector<float>& y, double* h3) {
    const double fund = bin_db(y, WARM, N, K0);
    *h3 = bin_db(y, WARM, N, 3 * K0) - fund;
    double worst = -400;
    for (int k = 8; k < (int)(19500.0 * N / FS); ++k) {
        bool nh = false;
        for (int m = 1; m * K0 < N / 2; ++m) if (std::abs(k - m * K0) <= 3) nh = true;
        if (nh) continue;
        worst = std::max(worst, bin_db(y, WARM, N, k) - fund);
    }
    return worst;
}

static void test_pipeline() {
    std::printf("pipeline\n");
    polydist::LineConnector d7;
    polydist::connector_between_lines(-1, 0, 1, 0, 7, d7);
    double worst_ref = 0, worst_ps = 0;
    for (float amp : {0.9f, 1.5f, 3.0f}) {
        std::vector<float> x = sine(amp), a(TOTAL), b(TOTAL), c(TOTAL);
        aadsp::Local4xPipeline<Cfg, true> ref(64);
        polydist::Shaper4x<Cfg> blk(64), ps(1);
        for (int i = 0; i < TOTAL; i += 64) ref.process_block(&x[i], &a[i], 64);
        for (int i = 0; i < TOTAL; i += 64) blk.process_block(&x[i], &b[i], 64, d7);
        for (int i = 0; i < TOTAL; ++i) ps.process_block(&x[i], &c[i], 1, d7);
        for (int i = 0; i < TOTAL; ++i) {
            worst_ref = std::max(worst_ref, (double)std::fabs(a[i] - b[i]));
            worst_ps = std::max(worst_ps, (double)std::fabs(b[i] - c[i]));
        }
    }
    std::printf("  max |Shaper4x - Local4xPipeline<clamped d7>| = %.2g, per-sample vs block = %.2g\n", worst_ref, worst_ps);
    // the factored evaluator and the compile-time monomial d7 differ by
    // float rounding of the same polynomial (~1e-5 through the pipeline)
    check(worst_ref < 5e-5, "Shaper4x with the d7 connector reproduces the verified lowering (rounding only)");
    check(worst_ps < 2e-6, "per-sample == block processing");
}

static void alias_table() {
    std::printf("alias floor of the 4x scaffold (worst non-harmonic bin < 19.5 kHz, dBc; 4.4 kHz sine)\n");
    std::printf("  %-28s", "lines / degree cap:");
    for (int cap : {1, 7, 15, 31, 63, 127, 255}) std::printf(" D=%-4d", cap);
    std::printf("\n");
    struct Case { const char* name; double lv, ls, rv, rs; float amp; };
    const Case cases[] = {
        {"smoothstep lines, drive 0.9", -1, 0, 1, 0, 0.9f},
        {"smoothstep lines, drive 1.5", -1, 0, 1, 0, 1.5f},
        {"smoothstep lines, drive 3.0", -1, 0, 1, 0, 3.0f},
        {"slopes 0.3/0.1, drive 1.5", -1, 0.3, 1, 0.1, 1.5f},
        {"asym -0.7/1.25 1.4/-0.35, 1.5", -0.7, 1.25, 1.4, -0.35, 1.5f},
    };
    for (const Case& cs : cases) {
        std::printf("  %-28s", cs.name);
        for (int cap : {1, 7, 15, 31, 63, 127, 255}) {
            polydist::LineConnector c;
            polydist::connector_between_lines(cs.lv, cs.ls, cs.rv, cs.rs, cap, c);
            std::vector<float> x = sine(cs.amp), y(TOTAL);
            polydist::Shaper4x<Cfg> s(64);
            for (int i = 0; i < TOTAL; i += 64) s.process_block(&x[i], &y[i], 64, c);
            double h3; std::printf(" %6.1f", worst_alias(y, &h3));
        }
        std::printf("\n");
    }
}

static void test_lut() {
    std::printf("LUT core (FAST mode)\n");
    // exact derivative vs central difference
    polydist::LineConnector c;
    polydist::connector_between_lines(-2, 5, 2, -5, 31, c);
    double dworst = 0;
    for (int i = 1; i < 200; ++i) {
        double x = -1.0 + 2.0 * i / 200, h = 1e-6;
        dworst = std::max(dworst, std::fabs(c.interiorDerivative(x) - (c(x + h) - c(x - h)) / (2 * h)));
    }
    check(dworst < 1e-7, "interiorDerivative matches central differences");
    // approximation at worst-case knobs across the degree range
    double aworst = 0;
    for (int cap : {1, 7, 63, 255}) {
        polydist::connector_between_lines(-2, 5, 2, -5, cap, c);
        polydist::LutCore lut;
        lut.beginBuild(c);
        while (!lut.step(7)) {}                    // amortized path
        for (int i = 0; i <= 65536; ++i) {
            double x = -1.2 + 2.4 * i / 65536;     // incl. exterior
            aworst = std::max(aworst, std::fabs((double)lut.table().eval((float)x) - c(x)));
        }
    }
    std::printf("  max |LUT - exact| over caps {1,7,63,255}, hard knobs: %.2g (%.1f dB)\n",
                aworst, 20 * std::log10(aworst / 7.0));
    check(aworst < 2.5e-6, "LUT at the float-table floor across the degree range (~1.5e-6 for the +-7 value range)");
    // join exactness: boundary knots carry the exact endpoint values
    polydist::connector_between_lines(-0.9, 0.4, 1.2, -0.15, 63, c);
    polydist::LutCore lut;
    lut.beginBuild(c);
    while (!lut.step(64)) {}
    check(std::fabs(lut.table().eval(-1.f) + 0.9f) < 1e-6
          && std::fabs(lut.table().eval(1.f) - 1.2f) < 1e-6,
          "joins exact at the boundary knots");
}

int main() {
    std::printf("LG57 self-test (48 kHz)\n");
    test_connector();
    test_lut();
    test_pipeline();
    alias_table();
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
