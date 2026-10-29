#ifndef WAVE_BASELINE_HPP
#define WAVE_BASELINE_HPP

#include <cstddef>
#include <cmath>

namespace wave_baseline {

inline std::size_t idx(std::size_t i, std::size_t j, std::size_t n) {
    return i * n + j;
}

/**
 * Executes the second-order Taylor expansion startup step:
 * u^1 = u^0 + 0.5 * lambda^2 * laplacian(u^0)
 */
inline void startup_step(
    const double* __restrict__ u0,
    double* __restrict__ u1,
    std::size_t n,
    double lambda_sq
) {
    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            double laplacian = u0[idx(i + 1, j, n)] + u0[idx(i - 1, j, n)]
                             + u0[idx(i, j + 1, n)] + u0[idx(i, j - 1, n)]
                             - 4.0 * u0[idx(i, j, n)];
            u1[idx(i, j, n)] = u0[idx(i, j, n)] + 0.5 * lambda_sq * laplacian;
        }
    }
}

/**
 * Executes a single interior Leapfrog time-stepping sweep:
 * u^{n+1} = 2*u^n - u^{n-1} + lambda^2 * laplacian(u^n)
 */
inline void leapfrog_step(
    const double* __restrict__ u_prev,
    const double* __restrict__ u_curr,
    double* __restrict__ u_next,
    std::size_t n,
    double lambda_sq
) {
    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            double laplacian = u_curr[idx(i + 1, j, n)] + u_curr[idx(i - 1, j, n)]
                             + u_curr[idx(i, j + 1, n)] + u_curr[idx(i, j - 1, n)]
                             - 4.0 * u_curr[idx(i, j, n)];
            u_next[idx(i, j, n)] = 2.0 * u_curr[idx(i, j, n)] - u_prev[idx(i, j, n)] + lambda_sq * laplacian;
        }
    }
}

} // namespace wave_baseline

#endif // WAVE_BASELINE_HPP