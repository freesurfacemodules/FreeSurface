#pragma once
// Polynomial connectors between two exterior lines by two-point Hermite
// interpolation (C++ port of design/line_connector.py, the reference).
//
//   left line   L(x) = left_value  + left_slope  * (x - xL)   for x < xL
//   interior    P(x), the unique minimum-degree polynomial matching
//               P^(k)(xL) = L^(k)(xL), P^(k)(xR) = R^(k)(xR), k = 0..r
//   right line  R(x) = right_value + right_slope * (x - xR)   for x > xR
//
// With a degree cap D the matchable smoothness is r = floor((D - 1) / 2)
// and the connector has degree 2r + 1 (an even cap adds nothing: the
// extra coefficient would be free, and the minimum-degree solution is
// chosen).  Derivatives of order >= 2 of a line vanish, so the endpoint
// conditions are (value, slope, 0, 0, ...).
//
// Because the exteriors are lines, every endpoint condition beyond value
// and slope is zero, so the connector is always a combination of four
// fixed Hermite bases per smoothness order r, each with closed-form
// positive binomial coefficients (u = 1 - t):
//
//     B_r(t) = t^(r+1) * sum_{j<=r}   C(r+j, j) u^j     (value at right)
//     D_r(t) = -u t^(r+1) * sum_{j<r} C(r+j, j) u^j     (slope at right)
//
// and A_r, C_r their mirrors.  Nothing is solved at runtime: a parameter
// change stores the knob values, and evaluation mixes the four factored
// bases read from a static binomial table.  The factored form has no
// cancellation except the final 4-term mix, so float evaluation stays at
// ~1e-5 absolute out to degree 63, where the former monomial Horner
// collapsed beyond degree ~16 (log 92.1).  The general Hermite solver
// below is kept as the self-test's low-degree cross-check reference.
#include <algorithm>
#include <cmath>

namespace polydist {

constexpr int kMaxDegreeCap = 256;           // C^127 connector (degree 255)
constexpr int kMaxSmoothness = (kMaxDegreeCap - 1) / 2;            // 127
// Above this smoothness order the audio path evaluates in double: the
// binomial row reaches C(2r, r), which passes float range (3.4e38) at
// r = 67, and the t^(r+1) power factors go subnormal well before the
// series terms stop mattering.  r = 63 (degree cap 127) is a verified
// clean bound for the float path; the double path is exact to ~4e-14 at
// degree 255 (checked against exact rational evaluation, log 92.3).
constexpr int kFloatSmoothnessLimit = 63;
constexpr int kMaxConditions = 16;  // reference solver only (cap 15)

struct Polynomial {
    int count = 0;                            // number of coefficients
    double c[kMaxConditions] = {};            // ascending powers of x
    float cf[kMaxConditions] = {};            // float copy for evaluation

    double operator()(double x) const {
        double y = 0.0;
        for (int k = count - 1; k >= 0; --k) y = y * x + c[k];
        return y;
    }
    float eval(float x) const {
        float y = 0.0f;
        for (int k = count - 1; k >= 0; --k) y = y * x + cf[k];
        return y;
    }
    // k-th derivative value at x (double), for verification
    double derivative(double x, int order) const {
        double d[kMaxConditions];
        int n = count;
        for (int k = 0; k < n; ++k) d[k] = c[k];
        for (int o = 0; o < order; ++o) {
            if (n <= 1) return 0.0;
            for (int k = 0; k + 1 < n; ++k) d[k] = (k + 1) * d[k + 1];
            --n;
        }
        double y = 0.0;
        for (int k = n - 1; k >= 0; --k) y = y * x + d[k];
        return y;
    }
    // Evaluation scale of the k-th derivative at x, sum_j |c_j| j!/(j-k)!
    // |x|^(j-k): the backward-error bound for Horner in double.
    double derivative_scale(double x, int order) const {
        double s = 0.0, f = 1.0;
        for (int j = order; j < count; ++j) {
            double p = 1.0;
            for (int m = j - order + 1; m <= j; ++m) p *= m;
            s += std::fabs(c[j]) * p * std::pow(std::fabs(x), j - order);
        }
        (void)f;
        return s;
    }
    void sync() { for (int k = 0; k < count; ++k) cf[k] = (float)c[k]; }
};

namespace detail {

inline double factorial(int n) {
    double f = 1.0;
    for (int k = 2; k <= n; ++k) f *= k;
    return f;
}

// Gaussian elimination with partial pivoting on the augmented matrix
// M (n x (n+1)).  Returns false if singular.
inline bool solve_augmented(int n, double M[kMaxConditions][kMaxConditions + 1],
                            double* x) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(M[r][col]) > std::fabs(M[pivot][col])) pivot = r;
        if (std::fabs(M[pivot][col]) < 1e-14) return false;
        if (pivot != col)
            for (int j = 0; j <= n; ++j) std::swap(M[col][j], M[pivot][j]);
        const double p = M[col][col];
        for (int j = col; j <= n; ++j) M[col][j] /= p;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double s = M[r][col];
            if (std::fabs(s) < 1e-18) continue;
            for (int j = col; j <= n; ++j) M[r][j] -= s * M[col][j];
        }
    }
    for (int i = 0; i < n; ++i) x[i] = M[i][n];
    return true;
}

// C(r+j, j) for j = 0..kMaxSmoothness, one row per smoothness order r.
// The float instantiation overflows to inf in rows above
// kFloatSmoothnessLimit; those rows are never read (eval dispatches to
// the double path there).
template <class T>
inline const T* binomialRow(int r) {
    static T table[kMaxSmoothness + 1][kMaxSmoothness + 1];
    static bool ready = false;
    if (!ready) {
        for (int rr = 0; rr <= kMaxSmoothness; ++rr) {
            double b = 1.0;
            for (int j = 0; j <= kMaxSmoothness; ++j) {
                table[rr][j] = (T)b;
                b = b * (rr + j + 1) / (j + 1);
            }
        }
        ready = true;
    }
    return table[r];
}

template <class T>
inline T seriesEval(const T* b, int m, T t) {  // sum_{j<=m} b[j] t^j
    T acc = b[m];
    for (int j = m - 1; j >= 0; --j) acc = acc * t + b[j];
    return acc;
}

template <class T>
inline T seriesDerivEval(const T* b, int m, T t) {  // d/dt of the above
    if (m < 1) return (T)0;
    T acc = m * b[m];
    for (int j = m - 1; j >= 1; --j) acc = acc * t + j * b[j];
    return acc;
}

template <class T>
inline T powInt(T t, int n) {  // exponentiation by squaring
    T p = (T)1;
    T b = t;
    while (n) {
        if (n & 1) p *= b;
        b *= b;
        n >>= 1;
    }
    return p;
}

} // namespace detail

// Unique minimum-degree polynomial with prescribed derivatives
// left[0..nL) at xL and right[0..nR) at xR (nL + nR <= kMaxConditions).
inline bool hermite_connector(const double* left, int nL,
                              const double* right, int nR,
                              double xL, double xR, Polynomial& out) {
    const int N = nL + nR;
    if (nL <= 0 || nR <= 0 || N > kMaxConditions || xR == xL) return false;
    const double h = xR - xL;
    // Normalized coordinate t = (x - xL) / h: Q^(k)(t) = h^k P^(k)(x).
    double M[kMaxConditions][kMaxConditions + 1] = {};
    int row = 0;
    for (int k = 0; k < nL; ++k, ++row) {          // t = 0: only t^k survives
        M[row][k] = detail::factorial(k);
        M[row][N] = left[k] * std::pow(h, k);
    }
    for (int k = 0; k < nR; ++k, ++row) {          // t = 1: j!/(j-k)!
        for (int j = k; j < N; ++j)
            M[row][j] = detail::factorial(j) / detail::factorial(j - k);
        M[row][N] = right[k] * std::pow(h, k);
    }
    double q[kMaxConditions];
    if (!detail::solve_augmented(N, M, q)) return false;
    // Q(t) = sum_j q[j] t^j -> P(x) = sum_k c[k] x^k via
    // ((x - xL)/h)^j = h^-j sum_k C(j,k) x^k (-xL)^(j-k)
    out.count = N;
    for (int k = 0; k < N; ++k) out.c[k] = 0.0;
    for (int j = 0; j < N; ++j) {
        const double hj = std::pow(h, j);
        for (int k = 0; k <= j; ++k) {
            const double binom = detail::factorial(j)
                / (detail::factorial(k) * detail::factorial(j - k));
            out.c[k] += q[j] * binom * std::pow(-xL, j - k) / hj;
        }
    }
    out.sync();
    return true;
}

struct LineConnector {
    double xL = -1.0, xR = 1.0;
    double left_value = -1.0, left_slope = 0.0;
    double right_value = 1.0, right_slope = 0.0;
    int degree_cap = 7;
    int smoothness_order = 3;                 // r; interior degree 2r + 1
    // float copies for the audio path
    float fxL = -1.f, fxR = 1.f, flv = -1.f, fls = 0.f, frv = 1.f, frs = 0.f;
    float fh = 2.f;                           // xR - xL

    int interior_degree() const { return 2 * smoothness_order + 1; }

    // exact interior dP/dx (double), by term-wise differentiation of the
    // factored bases (positive series throughout; used by the LUT build)
    double interiorDerivative(double x) const {
        const int r = smoothness_order;
        const double* b = detail::binomialRow<double>(r);
        const double h = xR - xL;
        const double t = (x - xL) / h, u = 1.0 - t;
        const double tr = detail::powInt(t, r), ur = detail::powInt(u, r);
        const double tp = tr * t, up = ur * u;
        const double St = detail::seriesEval(b, r, t);
        const double Su = detail::seriesEval(b, r, u);
        const double dSt = detail::seriesDerivEval(b, r, t);
        const double dSu = detail::seriesDerivEval(b, r, u);
        double acc = left_value * (-(r + 1) * ur * St + up * dSt)
                   + right_value * ((r + 1) * tr * Su - tp * dSu);
        if (r >= 1) {
            const double trm = detail::powInt(t, r - 1);
            const double urm = detail::powInt(u, r - 1);
            const double Ct = St - b[r] * tr, Cu = Su - b[r] * ur;
            const double dC = up * Ct - t * (r + 1) * ur * Ct
                + t * up * (dSt - r * b[r] * trm);
            const double dD = -tp * Cu + u * (r + 1) * tr * Cu
                - u * tp * (dSu - r * b[r] * urm);
            acc += h * (left_slope * dC - right_slope * dD);
        }
        return acc / h;
    }


    // factored four-basis evaluation (see header comment); T = float on
    // the audio path, double for the display / verification.  v^r comes
    // from exponentiation by squaring, S(r-1) from S(r) minus its top
    // term, and the arithmetic ordering matches the NEON path exactly.
    template <class T>
    T interiorEval(T x, T xl, T h, T lv, T ls, T rv, T rs) const {
        const int r = smoothness_order;
        const T* b = detail::binomialRow<T>(r);
        const T t = (x - xl) / h;
        const T u = (T)1 - t;
        const T tr = detail::powInt(t, r);
        const T ur = detail::powInt(u, r);
        const T tp = tr * t;
        const T up = ur * u;
        const T St = detail::seriesEval(b, r, t);
        const T Su = detail::seriesEval(b, r, u);
        T acc = lv * (up * St) + rv * (tp * Su);
        if (r >= 1) {
            const T C = (t * up) * (St - b[r] * tr);
            const T D = (u * tp) * (Su - b[r] * ur);
            acc += h * (ls * C - rs * D);
        }
        return acc;
    }

    float eval(float x) const {
        if (x < fxL) return flv + fls * (x - fxL);
        if (x > fxR) return frv + frs * (x - fxR);
        if (smoothness_order > kFloatSmoothnessLimit)   // see the constant
            return (float)interiorEval<double>((double)x, xL, xR - xL,
                                               left_value, left_slope,
                                               right_value, right_slope);
        return interiorEval<float>(x, fxL, fh, flv, fls, frv, frs);
    }
    // Deliberately scalar: the pipeline applies the core to four
    // independent rate-4 samples per base sample, whose evaluation chains
    // the out-of-order core already runs in parallel; a packed 4-lane
    // NEON version measured 50% slower than this form in the per-sample
    // pipeline (one serialized vector chain plus pack/unpack, log 92.2).
    void apply_block(float* out, const float* in, int n) const {
        for (int i = 0; i < n; ++i) out[i] = eval(in[i]);
    }

    double operator()(double x) const {
        if (x < xL) return left_value + left_slope * (x - xL);
        if (x > xR) return right_value + right_slope * (x - xR);
        return interiorEval<double>(x, xL, xR - xL, left_value, left_slope,
                                    right_value, right_slope);
    }
};

// Maximally smooth minimum-degree connector between two lines under a
// degree cap (connector_between_lines in the reference).  No solve: the
// knob values are stored and evaluation mixes the four bases.
inline bool connector_between_lines(double left_value, double left_slope,
                                    double right_value, double right_slope,
                                    int degree_cap, LineConnector& out,
                                    double xL = -1.0, double xR = 1.0) {
    degree_cap = std::max(1, std::min(kMaxDegreeCap, degree_cap));
    if (xR <= xL) return false;
    out.xL = xL; out.xR = xR;
    out.left_value = left_value; out.left_slope = left_slope;
    out.right_value = right_value; out.right_slope = right_slope;
    out.degree_cap = degree_cap;
    out.smoothness_order = (degree_cap - 1) / 2;
    out.fxL = (float)xL; out.fxR = (float)xR; out.fh = (float)(xR - xL);
    out.flv = (float)left_value; out.fls = (float)left_slope;
    out.frv = (float)right_value; out.frs = (float)right_slope;
    return true;
}

// Check the value and slope conditions of the factored evaluator at both
// joins (value directly, slope by central difference in double) and
// continuity with the exterior lines.  Higher-order conditions hold by
// construction of the closed-form bases (cross-checked against the
// reference solver in the self-test).
inline bool verify_line_connector(const LineConnector& c, double rel = 1e-6) {
    const double h = 1e-6 * (c.xR - c.xL);
    const double scale = 1.0 + std::fabs(c.left_value) + std::fabs(c.right_value)
        + std::fabs(c.left_slope) + std::fabs(c.right_slope);
    struct End { double x, v, s; };
    const End ends[2] = {{c.xL, c.left_value, c.left_slope},
                         {c.xR, c.right_value, c.right_slope}};
    for (const End& e : ends) {
        if (std::fabs(c(e.x) - e.v) > rel * scale) return false;
        if (c.smoothness_order < 1) continue;  // C0: no slope condition
        const double slope = (c(e.x + h) - c(e.x - h)) / (2 * h);
        if (std::fabs(slope - e.s) > 1e3 * rel * scale) return false;
    }
    return true;
}

} // namespace polydist
