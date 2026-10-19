#pragma once

#include <cstddef>
#include <vector>

namespace baseline {

inline std::size_t idx(const std::size_t i, const std::size_t j, const std::size_t n) {
    return i * n + j;
}

inline void gauss_seidel_poisson_sweep(
    std::vector<double>& psi,
    const std::vector<double>& omega,
    const std::size_t n,
    const double h,
    const std::size_t sweeps = 1) {
    const double h2 = h * h;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            for (std::size_t j = 1; j + 1 < n; ++j) {
                psi[idx(i, j, n)] = 0.25 * (
                    psi[idx(i + 1, j, n)] + psi[idx(i - 1, j, n)] +
                    psi[idx(i, j + 1, n)] + psi[idx(i, j - 1, n)] +
                    h2 * omega[idx(i, j, n)]);
            }
        }
    }
}

}  // namespace baseline
