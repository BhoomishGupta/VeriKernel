#pragma once
#include <cstddef>
#include <cmath>

namespace wave_baseline {

inline std::size_t idx(std::size_t i, std::size_t j, std::size_t n) {
    return i * n + j;
}

// INJECTED DEFECT: Replaces the 2nd-order Taylor expansion with a naive copy u1 = u0.
// This introduces an O(dt) error that collapses the global convergence order from p ~ 2.0 to p ~ 1.0.
inline void startup_step(
    const double* __restrict__ u0,
    double* __restrict__ u1,
    std::size_t n,
    double /*lambda_sq*/
) {
    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            u1[idx(i, j, n)] = u0[idx(i, j, n)]; // Injected Truncation Error!
        }
    }
}

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
